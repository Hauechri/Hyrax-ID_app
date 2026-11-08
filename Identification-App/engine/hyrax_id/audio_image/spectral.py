"""STFT based representations: linear / mel spectrogram (power or magnitude, dB or not)."""
from __future__ import annotations
import numpy as np
from .base import Param, Representation, YAxisInfo, resample_cols
from .registry import register


def stft_complex(audio, n_fft, hop, window):
    """Centered STFT, returns complex array (freq, time)."""
    pad = n_fft // 2
    x = np.pad(audio, (pad, pad), mode="reflect" if len(audio) > pad else "constant")
    win = np.hanning(n_fft) if window == "hann" else np.hamming(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * win, axis=1).T.astype(np.complex64)


def _stft_mag(audio, n_fft, hop, window):
    return np.abs(stft_complex(audio, n_fft, hop, window)).astype(np.float32)


def _hz_to_mel(f): return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)
def _mel_to_hz(m): return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - freqs) / max(hi - ce, 1e-9)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


def _delta(x, width=9):
    """Regression delta along time (axis 1)."""
    half = width // 2
    pad = np.pad(x, ((0, 0), (half, half)), mode="edge")
    num = sum(n * (pad[:, half + n: pad.shape[1] - half + n] - pad[:, half - n: pad.shape[1] - half - n])
              for n in range(1, half + 1))
    return (num / (2 * sum(n * n for n in range(1, half + 1)))).astype(np.float32)


class _SpectralBase(Representation):
    def _spec(self, audio, sr, n_cols):
        s = self.settings
        hop = max(1, len(audio) // n_cols)          # shared time axis
        mag = _stft_mag(audio, s["n_fft"], hop, s["window"])
        return mag ** 2 if s["power"] else mag

    def _finish(self, spec, n_cols):
        if self.settings["log"]:
            spec = 10 * np.log10(np.maximum(spec, 1e-10))
        spec = resample_cols(spec, n_cols)            # interpolate to exactly n_cols columns
        if self.settings.get("variant", "none") != "none":
            spec = _delta(spec)                        # delta's fixed window is tuned to this resolution
            if self.settings["variant"] == "delta-delta":
                spec = _delta(spec)
        return spec


_COMMON = [
    Param("n_fft", "pow2", 1024, "FFT size", 64, 16384),
    Param("window", "choice", "hann", "Window", choices=["hann", "hamming"]),
    Param("power", "bool", True, "Power (else magnitude)"),
    Param("log", "bool", True, "Log scale (dB)"),
    Param("variant", "choice", "none", "Variant", choices=["none", "delta", "delta-delta"]),
]


@register
class LinearSpectrogram(_SpectralBase):
    name, display_name = "linear_spec", "Spectrogram"
    params = _COMMON + [
        Param("fmin", "float", 0.0, "Min freq (Hz)", 0, 192000),
        Param("fmax", "float", 0.0, "Max freq (Hz, 0 = Nyquist)", 0, 192000),
    ]

    def _band(self, sr):
        fmax = self.settings["fmax"] or sr / 2
        return self.settings["fmin"], min(fmax, sr / 2)

    def y_axis(self, sr):
        lo, hi = self._band(sr)
        return YAxisInfo("hz", "Frequency", "Hz", lambda r: lo + (hi - lo) * np.asarray(r))

    def compute(self, audio, sr, n_cols):
        spec = self._spec(audio, sr, n_cols)
        freqs = np.linspace(0, sr / 2, spec.shape[0])
        lo, hi = self._band(sr)
        keep = (freqs >= lo) & (freqs <= hi)
        return self._finish(spec[keep], n_cols)


@register
class MelSpectrogram(_SpectralBase):
    name, display_name = "mel_spec", "Mel-Spectrogram"
    params = _COMMON + [
        Param("n_mels", "int", 128, "Mel bands", 8, 512),
        Param("fmin", "float", 0.0, "Min freq (Hz)", 0, 192000),
        Param("fmax", "float", 0.0, "Max freq (Hz, 0 = Nyquist)", 0, 192000),
    ]

    def _band(self, sr):
        return self.settings["fmin"], min(self.settings["fmax"] or sr / 2, sr / 2)

    def y_axis(self, sr):
        lo, hi = self._band(sr)
        def r2v(r):
            return _mel_to_hz(_hz_to_mel(lo) + (_hz_to_mel(hi) - _hz_to_mel(lo)) * np.asarray(r))
        return YAxisInfo("hz", "Frequency (mel spaced)", "Hz", r2v)

    def compute(self, audio, sr, n_cols):
        spec = self._spec(audio, sr, n_cols)
        lo, hi = self._band(sr)
        fb = mel_filterbank(sr, self.settings["n_fft"], self.settings["n_mels"], lo, hi)
        return self._finish(fb @ spec, n_cols)


@register
class PCENSpectrogram(MelSpectrogram):
    """Per-channel energy normalization on a mel spectrogram (robust to noise/gain)."""
    name, display_name = "pcen", "PCEN (mel)"
    params = [p for p in MelSpectrogram.params if p.name not in ("log", "power")] + [
        Param("time_constant", "float", 0.06, "Smoothing time (s)", 0.005, 2.0),
        Param("gain", "float", 0.98, "Gain (alpha)", 0.0, 1.0),
        Param("bias", "float", 2.0, "Bias (delta)", 0.0, 20.0),
        Param("power_exp", "float", 0.5, "Root (r)", 0.05, 1.0),
    ]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.settings.update(log=False, power=False)

    def compute(self, audio, sr, n_cols):
        s = self.settings
        hop = max(1, len(audio) // n_cols)
        mag = _stft_mag(audio, s["n_fft"], hop, s["window"])
        lo, hi = self._band(sr)
        E = mel_filterbank(sr, s["n_fft"], s["n_mels"], lo, hi) @ mag
        b = 1 - np.exp(-hop / (sr * s["time_constant"]))
        M = np.zeros_like(E)
        M[:, 0] = E[:, 0]
        for t in range(1, E.shape[1]):
            M[:, t] = (1 - b) * M[:, t - 1] + b * E[:, t]
        eps = 1e-6
        out = (E / (eps + M) ** s["gain"] + s["bias"]) ** s["power_exp"] - s["bias"] ** s["power_exp"]
        return resample_cols(out.astype(np.float32), n_cols)
