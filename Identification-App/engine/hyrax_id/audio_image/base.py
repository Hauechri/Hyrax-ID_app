"""Base classes: Param, YAxisInfo, Representation. No Qt imports allowed here."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional
import numpy as np


@dataclass
class Param:
    """Declarative parameter; the GUI builds its settings form from these."""
    name: str
    kind: str                      # "int" | "float" | "bool" | "choice" | "pow2"
    default: Any
    label: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[list] = None
    help: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.name.replace("_", " ").title()


@dataclass
class YAxisInfo:
    """Describes what the rows of a representation mean (for labels/axes)."""
    kind: str = "none"             # "hz" | "scale" | "index" | "amplitude" | "none"
    label: str = ""
    unit: str = ""
    # maps position in [0, 1] (0 = bottom row, 1 = top row) -> physical value; optional
    row_to_value: Optional[Callable[[np.ndarray], np.ndarray]] = None


class Representation:
    """One audio -> 2D array (rows, n_cols) transform.

    Convention: row 0 is the LOWEST y value. compose() flips for display.
    """
    name: ClassVar[str] = "base"
    display_name: ClassVar[str] = "Base"
    params: ClassVar[list[Param]] = []
    default_norm: ClassVar[str] = "percentile"   # see compose.NORM_MODES

    def __init__(self, **kwargs):
        defaults = {p.name: p.default for p in self.params}
        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {unknown}")
        self.settings: dict[str, Any] = {**defaults, **kwargs}

    def y_axis(self, sr: int) -> YAxisInfo:
        return YAxisInfo()

    def compute(self, audio: np.ndarray, sr: int, n_cols: int) -> np.ndarray:
        """Return float array of shape (rows, n_cols)."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"name": self.name, "params": dict(self.settings)}


def resample_cols(x: np.ndarray, n_cols: int) -> np.ndarray:
    """Linear interpolation along the time axis to exactly n_cols columns."""
    if x.shape[1] == n_cols:
        return x
    src = np.linspace(0.0, 1.0, x.shape[1])
    dst = np.linspace(0.0, 1.0, n_cols)
    return np.stack([np.interp(dst, src, row) for row in x]).astype(np.float32)


def resample_rows(x: np.ndarray, n_rows: int) -> np.ndarray:
    """Linear interpolation along the y axis to exactly n_rows rows."""
    if x.shape[0] == n_rows:
        return x
    return resample_cols(x.T, n_rows).T
