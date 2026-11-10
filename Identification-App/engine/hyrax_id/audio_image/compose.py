"""Channel specs -> RGB uint8 image. The time axis is validated, rows are resampled to H."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import registry
from .base import resample_rows

NORM_MODES = ["percentile", "minmax", "fixed", "none"]


@dataclass
class ChannelSpec:
    representation: str | None          # None -> zeros channel
    params: dict = field(default_factory=dict)
    norm: str | None = None             # None -> representation default
    norm_lo: float = 1.0                # percentile low / fixed min
    norm_hi: float = 99.0               # percentile high / fixed max

    def to_dict(self): return dict(self.__dict__)


@dataclass
class ImageSettings:
    sample_rate: int = 44100
    window_samples: int = 44100 * 3
    width: int = 640
    height: int = 640


def normalize(x, mode, lo, hi):
    if mode == "none":
        return np.clip(x, 0, 1)
    if mode == "percentile":
        a, b = np.percentile(x, [lo, hi])
    elif mode == "minmax":
        a, b = x.min(), x.max()
    else:  # fixed
        a, b = lo, hi
    return np.clip((x - a) / max(b - a, 1e-9), 0, 1)


def compute_channel(spec: ChannelSpec, audio, cfg: ImageSettings) -> np.ndarray:
    """Return (H, W) float in [0,1], row 0 = lowest y value."""
    if spec.representation is None:
        return np.zeros((cfg.height, cfg.width), dtype=np.float32)
    rep = registry.create(spec.representation, **spec.params)
    raw = rep.compute(audio, cfg.sample_rate, cfg.width)
    if raw.shape[1] != cfg.width:
        raise ValueError(f"{rep.name} returned {raw.shape[1]} columns, expected {cfg.width}")
    mode = spec.norm or rep.default_norm
    raw = normalize(raw, mode, spec.norm_lo, spec.norm_hi)
    return resample_rows(raw, cfg.height).astype(np.float32)


def fit_window(audio, n):
    return np.pad(audio, (0, n - len(audio))) if len(audio) < n else audio[:n]


def compose(audio, specs: list[ChannelSpec], cfg: ImageSettings):
    """Returns (rgb uint8 (H,W,3) with low y at bottom, list of per-channel (H,W) arrays)."""
    assert len(specs) == 3
    audio = fit_window(np.asarray(audio, dtype=np.float32), cfg.window_samples)
    chans = [compute_channel(s, audio, cfg) for s in specs]
    rgb = np.stack([np.flipud(c) for c in chans], axis=-1)
    return (rgb * 255).astype(np.uint8), chans
