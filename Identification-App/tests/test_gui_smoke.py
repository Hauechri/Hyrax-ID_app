"""Headless GUI test (Qt 'offscreen'): the preprocessing-config row, save/load, start validation and progress."""
import configparser
import json
import os
import shutil
import tempfile
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from conftest import CONFIGS

MEL = os.path.join(CONFIGS, "MELDELTAPCEN.json")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp, monkeypatch):
    import main_window as MW
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(("warning", a[2])))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(("critical", a[2])))
    w = MW.MainWindow()
    w._shown = shown
    yield w
    # Tear the window down while the QApplication is still alive. Leaving it to interpreter shutdown
    # (arbitrary destruction order of Qt objects) crashes intermittently, in the test process only.
    if w._thread is not None:
        w._thread.quit(); w._thread.wait()
    w.close()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()


def test_advanced_settings_no_longer_has_the_hardcoded_image_geometry(win):
    names = set(win._advanced_widgets)
    assert not names & {"window_size", "hop_size", "image_size", "isMulti"}
    assert {"window_overlap_percent", "detector_peak_normalize", "detector_threshold"} <= names


def test_summary_shows_what_the_config_means_or_why_it_fails(win, tmp_path):
    win.preprocess_edit.setText(MEL)
    t = win.preprocess_summary.text()
    assert "5 s windows" in t and "imgsz 980" in t and "R=mel_spec" in t and "B=pcen" in t
    win.preprocess_edit.setText(os.path.join(CONFIGS, "SigSpecPCEN_Strat.json"))
    assert "3 s windows" in win.preprocess_summary.text() and "R=waveform" in win.preprocess_summary.text()
    bad = tmp_path / "x.json"; bad.write_text(json.dumps({"dataset": {}}))
    win.preprocess_edit.setText(str(bad))
    assert "audio_image" in win.preprocess_summary.text()
    win.preprocess_edit.setText("")
    assert "No config chosen" in win.preprocess_summary.text()


def test_choosing_weights_picks_up_the_matching_config_and_its_overlap(win, tmp_path, monkeypatch):
    run = tmp_path / "run"; (run / "weights").mkdir(parents=True)
    d = json.load(open(MEL)); d["inference"]["overlap_percent"] = 75.0
    (run / "animalspot_config.json").write_text(json.dumps(d))
    weights = run / "weights" / "best.pt"; weights.write_bytes(b"w")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(weights), "")))
    win.preprocess_edit.setText("")
    win._browse_detector_path(win._model_path_edits["detector"])
    assert win._model_path_edits["detector"].text() == str(weights)
    assert win.preprocess_edit.text() == str(run / "animalspot_config.json")
    assert win._advanced_widgets["window_overlap_percent"].value() == 75.0
    assert "found next to the weights" in win.status_label.text()


def test_empty_field_is_ok_when_a_config_sits_next_to_the_weights(win, tmp_path):
    run = tmp_path / "r"; run.mkdir()
    shutil.copy(MEL, run / "animalspot_config.json"); (run / "best.pt").write_bytes(b"w")
    win._model_path_edits["detector"].setText(str(run / "best.pt"))
    win.preprocess_edit.setText("")
    assert "will be used" in win.preprocess_summary.text()


def test_settings_roundtrip_and_old_cfg_files_still_load(win, tmp_path, monkeypatch):
    win.preprocess_edit.setText(MEL)
    win._advanced_widgets["window_overlap_percent"].setValue(25.0)
    win._advanced_widgets["detector_peak_normalize"].setChecked(False)
    out = str(tmp_path / "s.cfg")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (out, "")))
    win._save_config()
    saved = configparser.ConfigParser(); saved.read(out)
    assert saved["hyraxid"]["preprocess_cfg_path"] == MEL and saved["hyraxid"]["window_overlap_percent"] == "25.0"
    win.preprocess_edit.setText(""); win._advanced_widgets["window_overlap_percent"].setValue(50.0)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (out, "")))
    win._load_config()
    assert win.preprocess_edit.text() == MEL
    assert win._advanced_widgets["window_overlap_percent"].value() == 25.0
    assert win._advanced_widgets["detector_peak_normalize"].isChecked() is False

    old = tmp_path / "old.cfg"                                    # a .cfg from before this change
    old.write_text("[hyraxid]\nwindow_size = 48000\nhop_size = 24000\nimage_size = 800\nismulti = True\n"
                   "detector_threshold = 0.4\n")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(old), "")))
    win._load_config()
    assert win._advanced_widgets["detector_threshold"].value() == 0.4
    assert win.preprocess_edit.text() == MEL                      # untouched: the old file has no such key


def _ready_for_start(win, tmp_path, preprocess):
    for key in ("denoise", "classify"):
        win._stage_checkboxes[key].setChecked(False)
    for prefix in ("detector", "garbage_filter"):
        f = tmp_path / f"{prefix}.bin"; f.write_bytes(b"m")
        win._model_path_edits[prefix].setText(str(f))
    wav = tmp_path / "a.wav"; wav.write_bytes(b"x")
    win.input_path_edit.setText(str(wav)); win.output_path_edit.setText(str(tmp_path / "out"))
    win.preprocess_edit.setText(preprocess)


def test_start_is_blocked_without_a_usable_config(win, tmp_path):
    _ready_for_start(win, tmp_path, "")
    win._on_start()
    assert win._thread is None and "settings file" in win._shown[-1][1].lower()
    bad = tmp_path / "bad.json"; bad.write_text("{}")
    win.preprocess_edit.setText(str(bad)); win._on_start()
    assert win._thread is None and "audio_image" in win._shown[-1][1]


def test_start_passes_the_config_to_the_pipeline_and_updates_progress(win, tmp_path, monkeypatch):
    import worker
    seen = {}

    def fake_run_batch(**kw):
        seen.update(kw)
        print("  - Windows 0/48")
        print("  - Windows 16/48")
        print("a normal log line")

    monkeypatch.setattr(worker, "run_batch", fake_run_batch)
    _ready_for_start(win, tmp_path, MEL)
    progress = []
    win._on_start()
    win._worker.slice_progress.connect(lambda c, t: progress.append((c, t)))
    deadline = time.time() + 10
    while win._thread is not None and time.time() < deadline:
        QCoreApplication.processEvents(); time.sleep(0.01)
    assert win._thread is None, "worker did not finish"
    assert seen["preprocess_cfg_path"] == MEL and seen["window_overlap_percent"] == 50.0
    assert seen["detector_peak_normalize"] is True
    assert not set(seen) & {"window_size", "hop_size", "image_size", "isMulti"}
    assert "a normal log line" in win.log_console.toPlainText()
    assert "Windows" not in win.log_console.toPlainText()         # progress lines go to the bar, not the log
    assert win.status_label.text() == "Done."


def test_progress_regex_follows_the_detector_output():
    import worker
    assert worker.SLICE_PROGRESS_RE.match("  - Windows 16/48").groups() == ("16", "48")
    assert worker.SLICE_PROGRESS_RE.match("  - Windows 0/48").groups() == ("0", "48")
    assert not worker.SLICE_PROGRESS_RE.match("  - Slice 1/48")


def test_standardize_settings_exist_and_follow_their_checkbox(win):
    w = win._advanced_widgets
    assert w["standardize_audio"].isChecked() and w["target_sample_rate"].value() == 48000
    assert w["audio_workers"].value() == 4
    w["audio_workers"].setValue(0)
    assert w["audio_workers"].text() == "auto"
    assert w["target_sample_rate"].isEnabled() and w["audio_workers"].isEnabled()
    w["standardize_audio"].setChecked(False)
    assert not w["target_sample_rate"].isEnabled() and not w["audio_workers"].isEnabled()
    w["standardize_audio"].setChecked(True)
    assert w["target_sample_rate"].isEnabled()


def test_start_passes_the_standardize_settings(win, tmp_path, monkeypatch):
    import worker
    seen = {}
    monkeypatch.setattr(worker, "run_batch", lambda **kw: seen.update(kw))
    _ready_for_start(win, tmp_path, MEL)
    win._advanced_widgets["target_sample_rate"].setValue(44100)
    win._advanced_widgets["audio_workers"].setValue(0)
    win._on_start()
    deadline = time.time() + 10
    while win._thread is not None and time.time() < deadline:
        QCoreApplication.processEvents(); time.sleep(0.01)
    assert seen["standardize_audio"] is True and seen["target_sample_rate"] == 44100 and seen["audio_workers"] == 0
