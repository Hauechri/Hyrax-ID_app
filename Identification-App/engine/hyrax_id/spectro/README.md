# spectro  (Spectrogram viewer engine)

Qt-free engine of the "Spectrogram viewer" tab: per-element Praat measurements (F0 range, F0 mean/median,
F1-F3), bout spectrograms and the CSV / PNG export. Input = the Audacity label files written by the detection
tab (`<recording>_labels.txt`) plus the recordings.

| file | origin |
|---|---|
| `praat_measures.py` | **verbatim copy** of `Spectrogram-Viewer/app/praat_measures.py` (do not edit here; re-copy it when the viewer prototype changes, so the measurements stay identical) |
| `library.py` | port of `spectrogram_core.py`: same rendering, measurements, CSV columns, file names and batch logic; the Excel annotation file and the built-in call detection are replaced by the detection tab's labels |

`tests/test_spectro.py` compares this engine with the prototype (set `SPECTROGRAM_VIEWER_DIR` to its `app` folder).

## Using it

GUI: second tab "Spectrogram viewer" (`app/spectrogram_tab.py`). Pick the recordings folder and the results folder
of the detection tab (or press "Take from Detection tab"); the tab's settings are saved in the same `.cfg` file.

Code:

    from hyrax_id.spectro.library import ResultsLibrary, RenderSettings
    lib = ResultsLibrary(audio_dir, labels_dir, output_dir)
    lib.run_batch([(a, r) for a in lib.list_animals() for r in lib.list_recordings(a)], RenderSettings())

`run_batch` uses spawned worker processes: call it from code guarded by `if __name__ == "__main__":`.
Extra packages: `requirements-spectrogram.txt`. Nuitka build: add `--include-package=parselmouth` and the matplotlib plugin.
