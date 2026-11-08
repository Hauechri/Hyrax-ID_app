"""Debug helper: saves the exact images that go to the detector."""
from __future__ import annotations
import itertools
from pathlib import Path

import numpy as np


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
