"""
Drop-in replacement for generate_input_image() in HYRAX_ID_predict.py.
Reproduces core/audio_image (compose + spectral) for MELDELTAPCEN.json:
  R = log power mel | G = delta of log power mel | B = PCEN mel
  sr 48000, window 240000 samples (5 s), 980x980, percentile 1-99 norm, low freq at BOTTOM.
"""
import numpy as np
from itertools import count
from pathlib import Path
from PIL import Image
DEBUG_SAVE = True
DEBUG_DIR = Path("D:/Hyrax-Identify/Base/Heloise_2024-2026/TMP/")
_IMAGE_COUNTER = count(1)

SR = 48000
WINDOW_SAMPLES = 240000
WIDTH = HEIGHT = 980
N_FFT, N_MELS, DELTA_WIDTH = 1024, 128, 9
PCEN_TC, PCEN_GAIN, PCEN_BIAS, PCEN_POWER = 0.06, 0.98, 2.0, 0.5
NORM_LO, NORM_HI = 1.0, 99.0


def stft_mag(audio, n_fft, hop):
    pad = n_fft // 2
    x = np.pad(audio, (pad, pad), mode="reflect" if len(audio) > pad else "constant")
    win = np.hanning(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.abs(np.fft.rfft(x[idx] * win, axis=1).T.astype(np.complex64)).astype(np.float32)


def _hz2mel(f): return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)
def _mel2hz(m): return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = _mel2hz(np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - freqs) / max(hi - ce, 1e-9)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


def delta(x, width=9):
    half = width // 2
    pad = np.pad(x, ((0, 0), (half, half)), mode="edge")
    num = sum(n * (pad[:, half + n: pad.shape[1] - half + n] - pad[:, half - n: pad.shape[1] - half - n])
              for n in range(1, half + 1))
    return (num / (2 * sum(n * n for n in range(1, half + 1)))).astype(np.float32)


def resample_cols(x, n):
    if x.shape[1] == n:
        return x
    src, dst = np.linspace(0, 1, x.shape[1]), np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, r) for r in x]).astype(np.float32)


def resample_rows(x, n):
    return x if x.shape[0] == n else resample_cols(x.T, n).T


def pcen(E, sr, hop):
    b = 1 - np.exp(-hop / (sr * PCEN_TC))
    M = np.zeros_like(E)
    M[:, 0] = E[:, 0]
    for t in range(1, E.shape[1]):
        M[:, t] = (1 - b) * M[:, t - 1] + b * E[:, t]
    out = (E / (1e-6 + M) ** PCEN_GAIN + PCEN_BIAS) ** PCEN_POWER - PCEN_BIAS ** PCEN_POWER
    return out.astype(np.float32)


def norm_percentile(x):
    a, b = np.percentile(x, [NORM_LO, NORM_HI])
    return np.clip((x - a) / max(b - a, 1e-9), 0, 1)


def generate_input_image(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """y: mono float32 in [-1, 1] ALREADY at 48 kHz (see load_audio_for_model). Returns (980, 980, 3) uint8 RGB."""
    if sr != SR:
        raise ValueError(f"Audio must be {SR} Hz, got {sr}. Resample first.")
    y = np.asarray(y, dtype=np.float32)
    # fit to exactly 5 s (pad with zeros / truncate) -> hop is identical to training
    y = np.pad(y, (0, WINDOW_SAMPLES - len(y))) if len(y) < WINDOW_SAMPLES else y[:WINDOW_SAMPLES]

    hop = max(1, len(y) // WIDTH)                                  # 244
    mag = stft_mag(y, N_FFT, hop)
    fb = mel_filterbank(sr, N_FFT, N_MELS, 0.0, sr / 2)

    # R: log power mel
    red = resample_cols(10 * np.log10(np.maximum(fb @ mag ** 2, 1e-10)), WIDTH)
    # G: delta of R (after column resampling)
    green = delta(red, DELTA_WIDTH)
    # B: PCEN on magnitude mel
    blue = resample_cols(pcen(fb @ mag, sr, hop), WIDTH)

    # normalise on the 128 x 980 array (BEFORE row resampling), then stretch rows to 980
    chans = [resample_rows(norm_percentile(c), HEIGHT).astype(np.float32) for c in (red, green, blue)]
    # low frequency at the BOTTOM of the image
    rgb = np.stack([np.flipud(c) for c in chans], axis=-1)
    image = (rgb * 255).astype(np.uint8)

    if DEBUG_SAVE:
        from PIL import Image
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(DEBUG_DIR / f"debug_{next(_IMAGE_COUNTER):05d}.png")

    return image


def load_audio_for_model(path, start_s=0.0):
    """Mono, float32 [-1,1], 48 kHz, 5 s segment (same as core/audio_io.load_segment)."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    with sf.SoundFile(path) as f:
        f.seek(min(int(start_s * f.samplerate), f.frames))
        x = f.read(int(WINDOW_SAMPLES / SR * f.samplerate) + 1, dtype="float32", always_2d=True)
        sr0 = f.samplerate
    x = x.mean(axis=1)
    if sr0 != SR:
        g = gcd(sr0, SR)
        x = resample_poly(x, SR // g, sr0 // g).astype(np.float32)
    return x