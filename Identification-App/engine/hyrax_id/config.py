"""Constants shared by the pipeline.

How the detector's input images are made (window length, image size, channels) is NOT defined here:
it comes from the YOLO app's settings file, see preprocess.PreprocessConfig.
"""
from __future__ import annotations

SR = 48000

SPEAKER_LIST = [
    'J9', 'Kashtan', 'M0', 'M9', 'O1', 'O7', 'P0', 'P1', 'P8', 'Q7',
    'R3', 'T0', 'T1', 'T9', 'U7', 'U9', 'W4', 'X0'
]
SPEAKER_TO_ID = {speaker: idx for idx, speaker in enumerate(sorted(SPEAKER_LIST))}
ID_TO_SPEAKER = {idx: speaker for speaker, idx in SPEAKER_TO_ID.items()}
