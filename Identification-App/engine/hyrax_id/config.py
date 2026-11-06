"""All settings and constants in one place.

DetectorImageConfig must describe EXACTLY how the detector's training images were made
(see MELDELTAPCEN.json). Window length, image size and imgsz all come from here, so they can
no longer drift apart between the audio->image step, the window slicing and the YOLO call.
"""
from __future__ import annotations
import json
from dataclasses import dataclass

SR = 48000

SPEAKER_LIST = [
    'J9', 'Kashtan', 'M0', 'M9', 'O1', 'O7', 'P0', 'P1', 'P8', 'Q7',
    'R3', 'T0', 'T1', 'T9', 'U7', 'U9', 'W4', 'X0'
]
SPEAKER_TO_ID = {speaker: idx for idx, speaker in enumerate(sorted(SPEAKER_LIST))}
ID_TO_SPEAKER = {idx: speaker for speaker, idx in SPEAKER_TO_ID.items()}


@dataclass(frozen=True)
class DetectorImageConfig:
    """Defaults = MELDELTAPCEN.json  (R: log power mel | G: delta of R | B: PCEN, percentile 1-99)."""
    sample_rate: int = 48000
    window_samples: int = 240000        # 5 s
    size: int = 992                     # width = height = YOLO imgsz
    n_fft: int = 1024
    n_mels: int = 128
    delta_width: int = 9
    pcen_time_constant: float = 0.06
    pcen_gain: float = 0.98
    pcen_bias: float = 2.0
    pcen_power: float = 0.5
    norm_lo: float = 1.0
    norm_hi: float = 99.0

    @property
    def window_s(self) -> float:
        return self.window_samples / self.sample_rate

    @classmethod
    def from_training_json(cls, path: str) -> "DetectorImageConfig":
        """Read the training UI settings file and refuse anything this renderer does not implement."""
        with open(path, encoding="utf-8") as f:
            ai = json.load(f)["audio_image"]
        img, ch = ai["image"], ai["channels"]
        layout = [(c["representation"], c["params"].get("variant", "none")) for c in ch]
        if layout != [("mel_spec", "none"), ("mel_spec", "delta"), ("pcen", "none")]:
            raise ValueError(f"Unsupported channel layout {layout}; audio2image.py only renders Mel / Mel-delta / PCEN.")
        if any(c["norm"] not in (None, "percentile") for c in ch):
            raise ValueError("Only percentile normalisation (norm null/'percentile') is implemented.")
        if img["width"] != img["height"]:
            raise ValueError("Non-square images are not supported by the YOLO wrapper.")
        m, p = ch[0]["params"], ch[2]["params"]
        return cls(sample_rate=img["sample_rate"], window_samples=img["window_samples"], size=img["width"],
                   n_fft=m["n_fft"], n_mels=m["n_mels"],
                   pcen_time_constant=p["time_constant"], pcen_gain=p["gain"],
                   pcen_bias=p["bias"], pcen_power=p["power_exp"],
                   norm_lo=ch[0]["norm_lo"], norm_hi=ch[0]["norm_hi"])
