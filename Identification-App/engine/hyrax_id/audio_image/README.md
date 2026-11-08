# audio_image  (vendored, do not edit here)

Verbatim copy of `TRAIN/core/audio_image/` from the YOLO training app (minus `presets.py`, which is GUI-only).
The detector's input images MUST be produced by this exact code, so when the YOLO app changes
(new representation, changed default, bug fix) copy these files over again:

    base.py  registry.py  compose.py  spectral.py  waveform.py  phase.py  cepstral.py  wavelet.py  __init__.py

`engine/tests/test_preprocess.py` re-renders every example config and fails if anything stops working.
