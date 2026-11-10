"""Waveform as image: min/max envelope filled into `rows` rows."""
from __future__ import annotations
import numpy as np
from .base import Param, Representation, YAxisInfo
from .registry import register


@register
class WaveformEnvelope(Representation):
    name, display_name = "waveform", "Waveform (envelope)"
    default_norm = "none"
    params = [
        Param("rows", "int", 256, "Rows", 16, 2048),
        Param("gain", "float", 1.0, "Amplitude gain", 0.1, 100.0),
    ]

    def y_axis(self, sr):
        return YAxisInfo("amplitude", "Amplitude", "", lambda r: np.asarray(r) * 2 - 1)

    def compute(self, audio, sr, n_cols):
        rows = self.settings["rows"]
        edges = np.linspace(0, len(audio), n_cols + 1).astype(int)
        img = np.zeros((rows, n_cols), dtype=np.float32)
        a = np.clip(audio * self.settings["gain"], -1, 1)
        for c in range(n_cols):
            seg = a[edges[c]:max(edges[c + 1], edges[c] + 1)]
            if seg.size == 0:
                continue
            lo = int((seg.min() + 1) / 2 * (rows - 1))
            hi = int((seg.max() + 1) / 2 * (rows - 1))
            img[lo:hi + 1, c] = 1.0
        return img
