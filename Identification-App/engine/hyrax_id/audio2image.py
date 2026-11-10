"""Audio window -> RGB image. Exact re-implementation of the training project's
core/audio_image (compose + spectral); verified pixel-identical against it.

  R = log power mel      G = delta of R (after column resampling)      B = PCEN (on magnitude mel)
  each channel: percentile (lo, hi) normalisation on the (n_mels x size) array, THEN rows stretched
  to `size`; image flipped so the LOWEST frequency is at the BOTTOM.
"""
from __future__ import annotations
import itertools
from pathlib import Path

import numpy as np

from .config import DetectorImageConfig


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


def pcen(E, sr, hop, cfg: DetectorImageConfig):
    b = 1 - np.exp(-hop / (sr * cfg.pcen_time_constant))
    M = np.zeros_like(E)
    M[:, 0] = E[:, 0]
    for t in range(1, E.shape[1]):
        M[:, t] = (1 - b) * M[:, t - 1] + b * E[:, t]
    out = (E / (1e-6 + M) ** cfg.pcen_gain + cfg.pcen_bias) ** cfg.pcen_power - cfg.pcen_bias ** cfg.pcen_power
    return out.astype(np.float32)


def _norm(x, cfg):
    a, b = np.percentile(x, [cfg.norm_lo, cfg.norm_hi])
    return np.clip((x - a) / max(b - a, 1e-9), 0, 1)


def generate_input_image(window: np.ndarray, cfg: DetectorImageConfig) -> np.ndarray:
    """window: mono float32 at cfg.sample_rate, exactly cfg.window_samples long (see windowing.get_window).
    Returns (size, size, 3) uint8 RGB."""
    if window.ndim != 1 or len(window) != cfg.window_samples:
        raise ValueError(f"Expected a mono window of {cfg.window_samples} samples, got shape {window.shape}.")
    sr, W = cfg.sample_rate, cfg.size
    hop = max(1, len(window) // W)
    mag = stft_mag(window.astype(np.float32), cfg.n_fft, hop)
    fb = mel_filterbank(sr, cfg.n_fft, cfg.n_mels, 0.0, sr / 2)

    red = resample_cols(10 * np.log10(np.maximum(fb @ mag ** 2, 1e-10)), W)
    green = delta(red, cfg.delta_width)
    blue = resample_cols(pcen(fb @ mag, sr, hop, cfg), W)

    chans = [resample_rows(_norm(c, cfg), cfg.size).astype(np.float32) for c in (red, green, blue)]
    rgb = np.stack([np.flipud(c) for c in chans], axis=-1)
    return (rgb * 255).astype(np.uint8)


class DebugImageSaver:
    """Writes the exact images that go to YOLO, so they can be compared with the training dataset.
    Stops after `max_images` so a long recording does not fill the disk."""
    def __init__(self, folder, max_images: int = 50):
        self.folder, self.max_images, self._n = Path(folder), max_images, itertools.count(1)
        self.folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, image: np.ndarray, tag: str = ""):
        i = next(self._n)
        if i > self.max_images:
            return
        from PIL import Image
        Image.fromarray(image).save(self.folder / f"debug_{i:05d}{('_' + tag) if tag else ''}.png")
