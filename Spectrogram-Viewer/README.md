# Acoustic Spectrogram Viewer

Desktop app (PySide6) for viewing hyrax song bouts as spectrograms and
measuring every element: F0 range (HzMax / HzMin), F0 mean / median and
F1–F3 formants. Praat is used through the `praat-parselmouth` package, so no
Praat installation is needed. Works with or without an annotation file, for
one recording or a whole folder.

## Install

Python 3.10+.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python app\main.py
```

1. **Data**: choose the audio folder (`.wav`). Tick **Use annotation file** to
   take bouts and element times from an Excel annotation file (see
   [app/DATA_SETUP.md](app/DATA_SETUP.md)); untick it to detect calls
   automatically. Changes apply immediately (**Reload library** re-reads the
   folder after adding files).
2. **Output folder**: where images and CSVs are written, as
   `<output>/<Animal>/<Recording>/<Animal>_<Recording>_bout_NN.png` / `..._hz.csv`.
3. **Single recording** tab: pick Animal / Recording / Bout →
   **Generate spectrogram** → **Save image** / **Save CSV**. After the first
   Generate, changing the bout or any option redraws automatically, so a saved
   file always matches the options shown.
   **Batch (folder)** tab: pick an animal (or all) and the outputs
   (PNG per bout, CSV per recording, one combined `batch_<date>_hz.csv`) →
   **Run batch**. Runs on several CPU cores; options are locked while it runs.

No data is included in this repository.

## What the image shows

- **Filled boxes**: HzMin / HzMax per element, detected from the audio
  (with an annotation file you can show the annotated values instead).
- **Dashed outlines**: the annotated (hand-measured) values, for comparison.
- **Lines**: F0 (Praat pitch) and F1 / F2 (Praat Burg formants), inside each element.

## How HzMax / HzMin are detected

Mimics the manual measurement in Avisoft (FFT 256, Hamming, 50 % overlap):
HzMax = upper edge of the F0 trace at its highest point, HzMin = lower edge
at its lowest point. Praat pitch, octave-jump cleaning, a sub-harmonic check
against the spectrogram, then the band edges. Details in
[app/praat_measures.py](app/praat_measures.py).

Agreement with the hand-measured values on 889 annotated elements (tuned on
half the recordings, checked on the other half):

| Element | n | median \|error\| max | median \|error\| min |
|---|---|---|---|
| Chuck | 644 | ~95 Hz | ~130 Hz |
| Wail | 179 | ~140 Hz | ~185 Hz |
| Snort | 66 | ~215 Hz | ~140 Hz |

For scale: the hand-measured values are on Avisoft's 94 Hz frequency grid.
Known weak spot: in noisy recordings Praat sometimes locks onto a
sub-harmonic (F0/2, F0/3) for a whole element; there is no real energy
there, but the check can't always lift it, so HzMin (sometimes HzMax) comes
out too low. Purely energy-based alternatives (strongest pixel / energy blob)
were tested and matched the annotations worse overall.

## Automatic call detection (no annotation file)

Tonality-based (strongest band minus median band, against a rolling
baseline), calls grouped into bouts at pauses > 0.8 s. Against the annotated
elements: ~66–76 % found, start / end within ~10–20 ms; only ~30 % of
detections match an annotated element (partly calls that weren't annotated,
partly noise). Element type is not classified ("Call"). Treat it as a first
pass. Details in [app/call_detection.py](app/call_detection.py).

## Formants

Praat Burg, 3 formants, 0.03 s window, mean over the element. The formant
ceiling is set in the UI (default 7500 Hz; the reference measurements fit
~7500–10000 Hz depending on the recording).

## CSV columns

`singer, song_id, bout, element, time_start, time_end, boundaries
(annotation / detected), annotated_hzmax, annotated_hzmin, auto_hzmax,
auto_hzmin, f0_mean_hz, f0_median_hz, F1_hz, F2_hz, F3_hz, formant_ceiling_hz`

F0 columns are empty when Praat finds no voiced pitch in the element.

## Standalone .exe

`build_exe.ps1` builds a Windows executable with Nuitka
(`pip install nuitka` first).
