# Acoustic Spectrogram Viewer

Desktop app (PySide6) for viewing hyrax song bouts as spectrograms, with
per-element F0 range (HzMax / HzMin) detected automatically from the audio
and F1–F3 formants measured in Praat (through the `praat-parselmouth`
package, so no Praat installation is needed).

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

Choose an **Annotation file** (Excel) and an **Audio folder** (`.wav` files),
click **Load library**, pick Animal / Recording / Bout, then
**Generate spectrogram**. How the annotation file and audio folder are matched
is described in [app/DATA_SETUP.md](app/DATA_SETUP.md). No data is included in
this repository.

## What it shows

- **Filled boxes**: HzMin / HzMax per element, detected automatically
  (switch *HzMin / HzMax from* to "Annotation file" to show the Excel values instead).
- **Dashed outlines**: the annotated (hand-measured) values, for comparison.
- **Lines**: F0 (Praat pitch) and F1 / F2 (Praat Burg formants), drawn inside each element.

Element start / end times always come from the annotation file; only the
frequencies are detected.

## How HzMax / HzMin are detected

The method mimics the manual measurement in Avisoft (FFT 256, Hamming, 50 %
overlap): HzMax is the upper edge of the F0 trace at its highest point, HzMin
the lower edge at its lowest point. See the docstring of
[app/praat_measures.py](app/praat_measures.py) for the steps.

Agreement with the hand-measured values on 889 annotated elements
(parameters tuned on half the recordings, checked on the other half):

| Element | n | median \|error\| max | median \|error\| min |
|---|---|---|---|
| Chuck | 644 | ~95 Hz | ~130 Hz |
| Wail | 179 | ~140 Hz | ~185 Hz |
| Snort | 66 | ~215 Hz | ~140 Hz |

For scale: the hand-measured values are on Avisoft's 94 Hz frequency grid.

## Formants

Praat Burg, 3 formants, 0.03 s window, mean over the element. The formant
ceiling is set in the UI (default 7500 Hz; the reference measurements fit
~7500–10000 Hz depending on the recording).

## CSV export

**Export F0 + formants (CSV)…** writes one row per element of the current bout
to `<Animal>_<Recording>_bout_NN_hz.csv`:

`singer, song_id, bout, element, time_start, time_end, annotated_hzmax,
annotated_hzmin, auto_hzmax, auto_hzmin, f0_mean_hz, f0_median_hz, F1_hz,
F2_hz, F3_hz, formant_ceiling_hz`

F0 columns are empty when Praat finds no voiced pitch in the element.

## Standalone .exe

`build_exe.ps1` builds a Windows executable with Nuitka
(`pip install nuitka` first).
