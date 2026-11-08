"""Scalograms via continuous wavelet transform (Morlet, Ricker, Haar).

Scales are defined by *pseudo-frequency* (log spaced between fmin and fmax), so the y axis
is a frequency axis in Hz that is independent of Nyquist. Output magnitude is pooled to n_cols.
"""
from __future__ import annotations
import numpy as np
from .base import Param, Representation, YAxisInfo
from .registry import register


def _kernel(kind, f, sr, cycles):
    if kind == "morlet":
        sigma = cycles / (2 * np.pi * f)
        half = int(4 * sigma * sr)
        t = np.arange(-half, half + 1) / sr
        return (np.exp(2j * np.pi * f * t) * np.exp(-t ** 2 / (2 * sigma ** 2))) / (sigma * sr)
    if kind == "ricker":
        a = np.sqrt(2) / (2 * np.pi * f)
        half = int(5 * a * sr)
        t = np.arange(-half, half + 1) / sr
        return ((1 - (t / a) ** 2) * np.exp(-t ** 2 / (2 * a ** 2))) / (a * sr)
    T = max(2, int(round(sr / f)))                       # haar: one full period
    k = np.ones(T); k[T // 2:] = -1
    return k / T


@register
class Scalogram(Representation):
    name, display_name = "scalogram", "Scalogram (wavelet)"
    params = [
        Param("wavelet", "choice", "morlet", "Wavelet", choices=["morlet", "ricker", "haar"]),
        Param("n_scales", "int", 96, "Number of scales", 8, 512),
        Param("fmin", "float", 100.0, "Min freq (Hz)", 1, 192000),
        Param("fmax", "float", 0.0, "Max freq (Hz, 0 = Nyquist/2)", 0, 192000),
        Param("cycles", "float", 6.0, "Morlet cycles", 2, 20),
        Param("log", "bool", True, "Log amplitude (dB)"),
    ]

    def _freqs(self, sr):
        s = self.settings
        fmax = s["fmax"] or sr / 4
        fmin = min(s["fmin"], fmax * 0.99)
        return np.geomspace(fmin, fmax, s["n_scales"])

    def y_axis(self, sr):
        fr = self._freqs(sr)
        return YAxisInfo("hz", "Pseudo-frequency", "Hz", lambda r: fr[0] * (fr[-1] / fr[0]) ** np.asarray(r))

    def compute(self, audio, sr, n_cols):
        s = self.settings
        n = len(audio)
        edges = np.linspace(0, n, n_cols + 1).astype(int)
        out = np.zeros((s["n_scales"], n_cols), dtype=np.float32)
        for i, f in enumerate(self._freqs(sr)):
            k = _kernel(s["wavelet"], f, sr, s["cycles"])
            size = 1 << int(np.ceil(np.log2(n + len(k))))
            conv = np.fft.ifft(np.fft.fft(audio, size) * np.fft.fft(k, size))
            conv = np.abs(conv[len(k) // 2: len(k) // 2 + n])
            cs = np.concatenate([[0], np.cumsum(conv)])   # block mean pooling to n_cols
            w = np.maximum(edges[1:] - edges[:-1], 1)
            out[i] = (cs[np.minimum(edges[1:], n)] - cs[edges[:-1]]) / w
        if s["log"]:
            out = 20 * np.log10(np.maximum(out, 1e-8))
        return out
