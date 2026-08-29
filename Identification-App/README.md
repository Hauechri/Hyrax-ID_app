# Hyrax-ID App

A desktop app (PySide6 + Nuitka) that identifies individual rock hyraxes from audio recordings.

Wraps the Hyrax-ID inference pipeline (denoiser → element detector → false-positive
filter → bout grouping → ECAPA-TDNN classifier) in a GUI: pick a `.wav` file or a
folder of recordings, watch a live log while it processes, and see which animal(s)
were identified when it's done.

See `SETUP.md` for environment setup and build instructions.

## Status

Work in progress — see the implementation phases as they land.
