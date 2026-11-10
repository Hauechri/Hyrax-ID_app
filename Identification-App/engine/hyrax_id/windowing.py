"""Slicing a recording into the same 5 s windows the detector was trained on."""
import numpy as np


def window_starts(n_samples: int, window: int, hop: int) -> np.ndarray:
    """Start sample of every window; together they cover the whole recording.
    The last window may run past the end (zero-padded by get_window), like the end windows in training."""
    if n_samples <= window:
        return np.array([0])
    n = int(np.ceil((n_samples - window) / hop)) + 1
    return np.arange(n) * hop


def get_window(signal: np.ndarray, start: int, window: int) -> np.ndarray:
    seg = signal[start:start + window]
    return np.pad(seg, (0, window - len(seg))) if len(seg) < window else seg
