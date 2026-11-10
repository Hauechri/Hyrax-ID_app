"""YOLO detector stage: signal -> windows -> images -> boxes -> (start_s, end_s, class_name, conf).

Everything about the images (window length, size, channels) comes from a PreprocessConfig, i.e. from the
YOLO app's settings file, so the same code serves any detector the YOLO app can train."""
from __future__ import annotations
from math import gcd

import numpy as np

from .debug import DebugImageSaver
from .preprocess import PreprocessConfig
from .windowing import window_starts, get_window


class YoloDetector:
    def __init__(self, model_path, preprocess: PreprocessConfig, conf: float, iou: float = 0.7,
                 device: str | None = None, batch_size: int = 16, debug_saver: DebugImageSaver | None = None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.names = dict(self.model.names)
        self.pre = preprocess
        self.batch_size = batch_size
        self.debug_saver = debug_saver
        self._kw = dict(conf=conf, iou=iou, imgsz=preprocess.imgsz, verbose=False)
        if device:
            self._kw["device"] = device
        try:    # the image size the weights were trained at (stored in the checkpoint), for a sanity check
            self.trained_imgsz = (getattr(self.model, "overrides", None) or {}).get("imgsz")
        except Exception:
            self.trained_imgsz = None

    def _predict(self, images):
        from PIL import Image
        # PIL = RGB for Ultralytics (a raw numpy array would be treated as BGR -> R/B swapped)
        results = self.model.predict(source=[Image.fromarray(im) for im in images], **self._kw)
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append([])
                continue
            out.append([(float(x0), float(x1), float(c), int(k))
                        for (x0, _y0, x1, _y1), c, k in zip(b.xyxyn.cpu().numpy(),     # normalised, so any size works
                                                            b.conf.cpu().numpy(),
                                                            b.cls.cpu().numpy().astype(int))])
        return out

    def detect(self, signal: np.ndarray, sr: int, hop_samples: int):
        """Returns raw labels [(start_s, end_s, class_name, conf)] in recording time."""
        pre = self.pre
        if sr != pre.sample_rate:                       # the images assume pre.sample_rate
            from scipy.signal import resample_poly
            g = gcd(sr, pre.sample_rate)
            signal = resample_poly(signal, pre.sample_rate // g, sr // g).astype(np.float32)
        signal = np.asarray(signal, dtype=np.float32)
        duration = len(signal) / pre.sample_rate
        starts = window_starts(len(signal), pre.window_samples, hop_samples)

        labels = []
        print(f"  - Windows 0/{len(starts)}", end="\r")
        for i in range(0, len(starts), self.batch_size):
            chunk = starts[i:i + self.batch_size]
            images = [pre.render(get_window(signal, int(s), pre.window_samples)) for s in chunk]
            if self.debug_saver is not None:
                for s, im in zip(chunk, images):
                    self.debug_saver(im, tag=f"t{s / pre.sample_rate:.1f}s")
            for s, boxes in zip(chunk, self._predict(images)):
                t0 = float(s) / pre.sample_rate
                for x0, x1, conf, k in boxes:
                    a, b = t0 + min(x0, x1) * pre.window_s, t0 + max(x0, x1) * pre.window_s
                    a, b = max(a, 0.0), min(b, duration)          # zero-padded tail windows can overshoot
                    if b > a:
                        labels.append((a, b, str(self.names.get(k, k)).lower(), conf))
            print(f"  - Windows {min(i + self.batch_size, len(starts))}/{len(starts)}", end="\r")
        print()
        return labels
