"""Detector preprocessing, driven entirely by the YOLO app's settings file.

The settings file (a "Save settings" export, or `animalspot_config.json` which the YOLO app stores
next to every trained model) is the single description of how the detector's training images were made:
audio window length, image size and the three channel recipes. This module loads it and renders windows
with the *same code* the YOLO app uses (hyrax_id/audio_image is a verbatim copy of its core/audio_image),
so any representation / parameter combination configured there works here without code changes.

What is taken from the file
    audio_image           -> sample rate, window length, image width/height, R/G/B channel recipes
    train.imgsz(_auto)    -> detector input size (max(width, height) when imgsz_auto, same rule as the YOLO app)
    dataset.box_mode      -> only time boxes are used by this pipeline (the y extent of a box is ignored)
    dataset.class_order   -> used to warn when the weights and the settings file do not belong together
    inference.overlap_percent / iou -> defaults (the confidence threshold is NOT taken: it is calibrated
                                       together with the garbage filter)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .audio_image import ChannelSpec, ImageSettings, compose, registry
from .audio_image.compose import NORM_MODES

SCHEMA_VERSION = 1                                  # same as the YOLO app's core.config.SCHEMA_VERSION
SETTINGS_FILENAME = "animalspot_config.json"        # file the YOLO app saves next to a trained model
DEFAULT_OVERLAP_PERCENT = 50.0
_STRIDE = 32                                        # Ultralytics rounds imgsz up to a multiple of the stride


class PreprocessConfigError(ValueError):
    """The settings file cannot be used. The message is written for the person using the GUI."""


def find_settings_for_weights(weights_path) -> Path | None:
    """The settings file that belongs to a weights file: same folder, else one folder above
    (Ultralytics layout: <run>/weights/best.pt with the settings in <run>). None if there is none."""
    if not weights_path:
        return None
    w = Path(weights_path)
    for folder in (w.parent, w.parent.parent):
        p = folder / SETTINGS_FILENAME
        if p.is_file():
            return p
    return None


def _round_up(n: int) -> int:
    return int(math.ceil(n / _STRIDE) * _STRIDE)


def _channel_label(spec: ChannelSpec) -> str:
    if spec.representation is None:
        return "empty"
    extra = [str(spec.params[k]) for k in ("variant", "delta", "component", "wavelet")
             if spec.params.get(k) not in (None, "none")]
    return spec.representation + (f"({', '.join(extra)})" if extra else "")


@dataclass
class PreprocessConfig:
    image: ImageSettings
    channels: list                          # three ChannelSpec (R, G, B)
    imgsz: int                              # detector input size
    box_mode: str = "time"
    class_order: tuple = ()
    overlap_percent: float | None = None    # from the file's inference section, if it has one
    iou: float | None = None
    source: str = ""

    # ---- geometry
    @property
    def sample_rate(self) -> int:
        return self.image.sample_rate

    @property
    def window_samples(self) -> int:
        return self.image.window_samples

    @property
    def window_s(self) -> float:
        return self.image.window_samples / self.image.sample_rate

    def hop_samples(self, overlap_percent: float | None = None) -> int:
        """Window hop in samples. overlap_percent=None -> the file's value, else 50 %."""
        if overlap_percent is None:
            overlap_percent = self.overlap_percent if self.overlap_percent is not None else DEFAULT_OVERLAP_PERCENT
        if not 0 <= overlap_percent < 100:
            raise PreprocessConfigError("Window overlap must be at least 0 and below 100 percent.")
        return max(1, int(round(self.window_samples * (1 - overlap_percent / 100))))

    # ---- rendering
    def render(self, window: np.ndarray) -> np.ndarray:
        """Mono float audio at self.sample_rate -> (height, width, 3) uint8 RGB, low y at the bottom.
        Shorter audio is zero-padded and longer audio truncated to the window (compose.fit_window)."""
        rgb, _ = compose(window, self.channels, self.image)
        return rgb

    @property
    def uses_absolute_amplitude(self) -> bool:
        """True if a channel's look depends on the absolute signal level (waveform envelope, no normalisation).
        The spectral channels are level-independent after their percentile normalisation."""
        return any(c.representation == "waveform" for c in self.channels)

    def describe(self) -> str:
        w, h = self.image.width, self.image.height
        ch = "  ".join(f"{n}={_channel_label(c)}" for n, c in zip("RGB", self.channels))
        return (f"{self.sample_rate / 1000:g} kHz, {self.window_s:g} s windows, {w}x{h} image, "
                f"imgsz {self.imgsz}, boxes: {self.box_mode}  |  {ch}")

    def check_model(self, class_names, trained_imgsz=None) -> list[str]:
        """Warnings if the weights do not look like they belong to this settings file."""
        out = []
        if self.class_order:
            want = {str(c).lower() for c in self.class_order}
            have = {str(c).lower() for c in class_names}
            if want != have:
                out.append(f"The settings file lists the classes {sorted(want)} but the weights contain "
                           f"{sorted(have)} -- are the weights and the settings file from the same training run?")
        if isinstance(trained_imgsz, (int, float)) and _round_up(int(trained_imgsz)) != _round_up(self.imgsz):
            out.append(f"The weights were trained at imgsz {int(trained_imgsz)} but the settings file gives "
                       f"imgsz {self.imgsz} -- are the weights and the settings file from the same training run?")
        return out

    # ---- loading
    @classmethod
    def from_dict(cls, data: dict, source: str = "") -> "PreprocessConfig":
        if not isinstance(data, dict) or "audio_image" not in data:
            raise PreprocessConfigError(
                "This file has no 'audio_image' section. Choose the settings file saved by the YOLO app "
                "(e.g. animalspot_config.json next to best.pt, or a 'Save settings' export).")
        if data.get("version", 0) > SCHEMA_VERSION:
            raise PreprocessConfigError("This settings file was written by a newer version of the YOLO app.")
        ai = data["audio_image"]
        try:
            img_d, ch_d = ai["image"], ai["channels"]
        except (KeyError, TypeError):
            raise PreprocessConfigError("The 'audio_image' section needs both 'image' and 'channels'.")

        ok_i = {f.name for f in fields(ImageSettings)}
        try:
            image = ImageSettings(**{k: v for k, v in img_d.items() if k in ok_i})
        except TypeError as e:
            raise PreprocessConfigError(f"Invalid 'image' settings: {e}")
        for name in ("sample_rate", "window_samples", "width", "height"):
            v = getattr(image, name)
            if not isinstance(v, int) or v <= 0:
                raise PreprocessConfigError(f"Image setting '{name}' must be a positive whole number (got {v!r}).")

        if len(ch_d) != 3:
            raise PreprocessConfigError(f"Exactly three channels (R, G, B) are needed, the file has {len(ch_d)}.")
        ok_c = {f.name for f in fields(ChannelSpec)}
        channels = []
        for letter, c in zip("RGB", ch_d):
            spec = ChannelSpec(**{k: v for k, v in c.items() if k in ok_c})
            if spec.representation is not None:
                if spec.representation not in registry.available():
                    raise PreprocessConfigError(
                        f"Channel {letter}: unknown representation '{spec.representation}'. This app knows: "
                        f"{', '.join(sorted(registry.available()))}. If the YOLO app has a newer one, copy its "
                        f"core/audio_image folder into engine/hyrax_id/audio_image.")
                try:
                    registry.create(spec.representation, **spec.params)    # rejects unknown parameter names
                except (ValueError, TypeError) as e:
                    raise PreprocessConfigError(f"Channel {letter}: {e}")
            if spec.norm is not None and spec.norm not in NORM_MODES:
                raise PreprocessConfigError(f"Channel {letter}: unknown normalisation '{spec.norm}'.")
            channels.append(spec)

        train = data.get("train") or {}
        imgsz = max(image.width, image.height) if train.get("imgsz_auto", True) else int(train.get("imgsz", 640))
        ds = data.get("dataset") or {}
        inf = data.get("inference") or {}
        classes = [ds.get("binary_name") or "target"] if ds.get("binary") else (ds.get("class_order") or [])
        return cls(image=image, channels=channels, imgsz=imgsz,
                   box_mode=ds.get("box_mode", "time"), class_order=tuple(classes),
                   overlap_percent=inf.get("overlap_percent"), iou=inf.get("iou"), source=source)

    @classmethod
    def from_file(cls, path) -> "PreprocessConfig":
        p = Path(path)
        if not p.is_file():
            raise PreprocessConfigError(f"Settings file not found: {p}")
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise PreprocessConfigError(f"Could not read {p.name}: {e}")
        return cls.from_dict(data, source=str(p))
