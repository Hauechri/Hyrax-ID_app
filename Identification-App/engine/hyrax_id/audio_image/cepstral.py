"""Cepstral representations: MFCC and LFCC (+ optional deltas)."""
from __future__ import annotations
import numpy as np
from .base import Param, Representation, YAxisInfo, resample_cols
from .registry import register
from .spectral import _stft_mag, mel_filterbank, _delta


def _dct_matrix(n_coeffs, n_in):
    n = np.arange(n_in)
    k = np.arange(n_coeffs)[:, None]
    m = np.cos(np.pi / n_in * (n + 0.5) * k) * np.sqrt(2.0 / n_in)
    m[0] /= np.sqrt(2)
    return m.astype(np.float32)


class _CepstralBase(Representation):
    def y_axis(self, sr):
        return YAxisInfo("index", "Cepstral coefficient", "", lambda r: np.asarray(r) * (self.settings["n_coeffs"] - 1))

    def _filterbank(self, sr):
        raise NotImplementedError

    def compute(self, audio, sr, n_cols):
        s = self.settings
        hop = max(1, len(audio) // n_cols)
        pw = _stft_mag(audio, s["n_fft"], hop, s["window"]) ** 2
        logfb = 10 * np.log10(np.maximum(self._filterbank(sr) @ pw, 1e-10))
        c = _dct_matrix(s["n_coeffs"], logfb.shape[0]) @ logfb
        if s["delta"] != "none":
            d = _delta(c)
            c = d if s["delta"] == "delta" else _delta(d)
        return resample_cols(c.astype(np.float32), n_cols)


_CEPS = [
    Param("n_fft", "pow2", 1024, "FFT size", 64, 16384),
    Param("window", "choice", "hann", "Window", choices=["hann", "hamming"]),
    Param("n_filters", "int", 64, "Filters", 8, 256),
    Param("n_coeffs", "int", 20, "Coefficients", 2, 128),
    Param("delta", "choice", "none", "Variant", choices=["none", "delta", "delta-delta"]),
]


@register
class MFCC(_CepstralBase):
    name, display_name = "mfcc", "MFCC"
    params = _CEPS + [Param("fmin", "float", 0.0, "Min freq (Hz)", 0, 192000),
                      Param("fmax", "float", 0.0, "Max freq (Hz, 0 = Nyquist)", 0, 192000)]

    def _filterbank(self, sr):
        s = self.settings
        return mel_filterbank(sr, s["n_fft"], s["n_filters"], s["fmin"], min(s["fmax"] or sr / 2, sr / 2))


@register
class LFCC(_CepstralBase):
    name, display_name = "lfcc", "LFCC"
    params = _CEPS

    def _filterbank(self, sr):
        s = self.settings
        n_bins = s["n_fft"] // 2 + 1
        edges = np.linspace(0, n_bins - 1, s["n_filters"] + 2)
        f = np.arange(n_bins)
        fb = np.zeros((s["n_filters"], n_bins), dtype=np.float32)
        for i in range(s["n_filters"]):
            lo, ce, hi = edges[i:i + 3]
            fb[i] = np.maximum(0, np.minimum((f - lo) / max(ce - lo, 1e-9), (hi - f) / max(hi - ce, 1e-9)))
        return fb
