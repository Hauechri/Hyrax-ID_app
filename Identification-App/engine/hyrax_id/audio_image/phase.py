"""Phase based representations from the complex STFT."""
from __future__ import annotations
import numpy as np
from .base import Param, Representation, YAxisInfo, resample_cols
from .registry import register
from .spectral import stft_complex


@register
class StftComplexPart(Representation):
    name, display_name = "stft_phase", "STFT phase / real / imag"
    default_norm = "none"
    params = [
        Param("component", "choice", "phase", "Component", choices=["phase", "real", "imag"]),
        Param("n_fft", "pow2", 1024, "FFT size", 64, 16384),
        Param("window", "choice", "hann", "Window", choices=["hann", "hamming"]),
    ]

    def y_axis(self, sr):
        return YAxisInfo("hz", "Frequency", "Hz", lambda r: np.asarray(r) * sr / 2)

    def compute(self, audio, sr, n_cols):
        s = self.settings
        hop = max(1, len(audio) // n_cols)
        Z = stft_complex(audio, s["n_fft"], hop, s["window"])
        if s["component"] == "phase":
            out = (np.angle(Z) + np.pi) / (2 * np.pi)          # -> [0, 1]
        else:
            part = Z.real if s["component"] == "real" else Z.imag
            out = 0.5 + 0.5 * np.tanh(part / (np.abs(part).mean() * 3 + 1e-9))   # squash -> [0, 1]
        return resample_cols(out.astype(np.float32), n_cols)
