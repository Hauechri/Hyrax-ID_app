"""Headless GUI tests (Qt 'offscreen') for the Spectrogram viewer tab and for settings shared with tab 1."""
import configparser
import os
import subprocess
import sys
import time

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("parselmouth")
pytest.importorskip("matplotlib")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from conftest import REAL_SOUNDFILE, ROOT

pytestmark = pytest.mark.skipif(not REAL_SOUNDFILE, reason="needs the real soundfile package")
import spectro_data as D


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return D.make_dataset(str(tmp_path_factory.mktemp("tabdata")))


@pytest.fixture
def win(qapp, monkeypatch):
    import main_window as MW
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(("warning", a[2])))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(("critical", a[2])))
    w = MW.MainWindow()
    w._shown = shown
    yield w
    w.spectrogram_tab.shutdown()
    if w._thread is not None:
        w._thread.quit(); w._thread.wait()
    w.close()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()


def _load(tab, dataset, output=None):
    audio, res = dataset
    tab.audio_row.edit.setText(audio)
    tab.labels_row.edit.setText(res)
    if output:
        tab.output_row.edit.setText(output)
    assert tab._load_library(interactive=False)


def _wait(cond, seconds=60):
    end = time.time() + seconds
    while not cond() and time.time() < end:
        QCoreApplication.processEvents(); time.sleep(0.02)
    assert cond(), "timed out"


# ------------------------------------------------------------------------------ structure
def test_the_window_has_both_tabs(win):
    assert win.tabs.count() == 2
    assert win.tabs.tabText(0) == "Detection / Identification" and win.tabs.tabText(1) == "Spectrogram viewer"
    assert win.spectrogram_tab.available
    assert win.centralWidget() is win.tabs


def test_library_populates_the_dropdowns(win, dataset, tmp_path):
    tab = win.spectrogram_tab
    _load(tab, dataset, str(tmp_path / "out"))
    animals = [tab.animal_combo.itemText(i) for i in range(tab.animal_combo.count())]
    assert animals == ["FB", "QY", "unknown"]
    assert [tab.batch_animal_combo.itemText(i) for i in range(tab.batch_animal_combo.count())][0] == "All animals"
    tab.animal_combo.setCurrentText("unknown")
    assert tab.recording_combo.currentText() == "rec2" and tab.bout_combo.count() == 2
    assert tab.bout_combo.itemText(0).startswith("1   (1.0–2.6 s)")
    s = tab.status_label.text()
    assert "4 bout(s) in 2 recording(s)" in s and "1 label file(s) without a recording" in s and "1 without any bout" in s


def test_no_classification_everything_is_one_animal(win, tmp_path, dataset):
    audio, res = dataset
    only = tmp_path / "only"; only.mkdir()
    (only / "rec2_labels.txt").write_text(open(os.path.join(res, "rec2_labels.txt"), encoding="utf-8").read(), encoding="utf-8")
    tab = win.spectrogram_tab
    tab.audio_row.edit.setText(audio); tab.labels_row.edit.setText(str(only))
    tab._load_library(interactive=False)
    assert tab.animal_combo.count() == 1 and tab.animal_combo.itemText(0) == "unknown"
    assert "no classification" in tab.status_label.text()


def test_wrong_folders_are_explained(win, dataset, tmp_path):
    tab = win.spectrogram_tab
    audio, res = dataset
    empty = tmp_path / "empty"; empty.mkdir()
    tab.audio_row.edit.setText(str(empty)); tab.labels_row.edit.setText(res)        # no recordings there
    assert tab._load_library(interactive=True)
    assert "No bouts found" in tab.status_label.text()
    assert "none has a recording of the same name" in win._shown[-1][1]
    tab.audio_row.edit.setText(audio); tab.labels_row.edit.setText(str(empty))      # no label files
    tab._load_library(interactive=True)
    assert "No *_labels.txt files" in win._shown[-1][1]
    tab.labels_row.edit.setText(str(tmp_path / "missing"))
    assert not tab._load_library(interactive=True) and "does not exist" in win._shown[-1][1]


# ------------------------------------------------------------------------------ output folder default
def test_output_folder_defaults_next_to_the_results_and_never_overrides_the_user(win, dataset, tmp_path):
    tab = win.spectrogram_tab
    audio, res = dataset
    tab.audio_row.edit.setText(audio); tab.labels_row.edit.setText(res)
    tab._load_library(interactive=False)
    assert tab.output_row.text() == os.path.join(res, "spectrograms")
    other = tmp_path / "res2"; other.mkdir()
    tab.labels_row.edit.setText(str(other)); tab._load_library(interactive=False)
    assert tab.output_row.text() == os.path.join(str(other), "spectrograms")      # our own default follows
    tab.output_row.edit.setText(str(tmp_path / "mine"))
    tab.labels_row.edit.setText(res); tab._load_library(interactive=False)
    assert tab.output_row.text() == str(tmp_path / "mine")                         # the user's choice stays


# ------------------------------------------------------------------------------ single recording
def test_generate_save_image_and_csv(win, dataset, tmp_path):
    tab = win.spectrogram_tab
    _load(tab, dataset, str(tmp_path / "out"))
    assert not tab.save_btn.isEnabled()
    tab.animal_combo.setCurrentText("FB")
    tab._on_generate()
    assert tab.last_render is not None and tab.save_btn.isEnabled() and tab.export_csv_btn.isEnabled()
    assert "3 element(s)" in tab.status_label.text()
    assert tab.fig.axes[0].get_title() == "Bout 1  |  Animal: FB  |  Recording: rec1"
    tab._on_save(); tab._on_export_csv()
    png = tmp_path / "out" / "FB" / "rec1" / "FB_rec1_bout_01.png"
    csv = tmp_path / "out" / "FB" / "rec1" / "FB_rec1_bout_01_hz.csv"
    assert png.is_file() and png.stat().st_size > 5000
    df = pd.read_csv(csv)
    assert list(df["element"]) == ["Chuck", "Wail", "Chuck"] and df["F1_hz"].notna().all()


def test_options_redraw_live_after_the_first_generate(win, dataset, tmp_path):
    tab = win.spectrogram_tab
    _load(tab, dataset, str(tmp_path / "out"))
    tab.animal_combo.setCurrentText("FB")
    tab._on_generate()
    tab.freq_max_spin.setValue(3000)
    assert tab._rerender_timer.isActive()
    tab._flush_rerender()
    assert tab.fig.axes[0].get_ylim()[1] >= 3000


# ------------------------------------------------------------------------------ batch
def test_batch_from_the_gui(win, dataset, tmp_path):
    tab = win.spectrogram_tab
    out = tmp_path / "out"
    _load(tab, dataset, str(out))
    tab.batch_workers_spin.setValue(1)            # in-process: no worker processes inside the test run
    tab.batch_animal_combo.setCurrentText("FB")
    assert "1 recording(s), 1 bout(s)" in tab.batch_summary.text()
    tab._on_run_batch()
    assert not tab.data_box.isEnabled() and tab.batch_cancel_btn.isEnabled()      # options locked while running
    _wait(lambda: tab._batch_thread is None or not tab._batch_thread.isRunning())
    QCoreApplication.processEvents()
    assert tab.data_box.isEnabled() and not tab.batch_cancel_btn.isEnabled()
    assert (out / "FB" / "rec1" / "FB_rec1_bout_01.png").is_file()
    assert (out / "FB" / "rec1" / "FB_rec1_hz.csv").is_file()
    log = tab.batch_log.toPlainText()
    assert "FB rec1 bout 1: 3 element(s)" in log and "Done: 1/1 bout(s)" in log
    assert "Batch done" in tab.status_label.text()


def test_batch_needs_an_output_and_a_selection(win, dataset, tmp_path):
    tab = win.spectrogram_tab
    tab._on_run_batch()
    assert "No recordings found" in win._shown[-1][1]
    _load(tab, dataset, str(tmp_path / "out"))
    for c in (tab.batch_png_check, tab.batch_rec_csv_check, tab.batch_all_csv_check):
        c.setChecked(False)
    tab._on_run_batch()
    assert "Tick at least one output" in win._shown[-1][1]


# ------------------------------------------------------------------------------ link to the detection tab
def test_take_paths_from_the_detection_tab(win, dataset, tmp_path):
    tab = win.spectrogram_tab
    audio, res = dataset
    win.input_path_edit.setText(os.path.join(audio, "rec2.wav"))       # single file -> its folder
    win.output_path_edit.setText(res)
    tab.take_btn.click()
    assert tab.audio_row.text() == audio and tab.labels_row.text() == res
    assert tab.animal_combo.count() == 3
    win.folder_radio.setChecked(True)
    win.input_path_edit.setText(audio)
    assert win._detection_paths() == (audio, res)


def test_finished_detection_run_leaves_a_hint(win):
    win.spectrogram_tab.data_hint.setText("x")
    win._on_finished()
    assert "detection run just finished" in win.spectrogram_tab.data_hint.text()


# ------------------------------------------------------------------------------ one .cfg for both tabs
def test_settings_file_holds_both_tabs(win, dataset, tmp_path, monkeypatch):
    tab = win.spectrogram_tab
    audio, res = dataset
    # tab 1
    win.input_path_edit.setText(audio); win.output_path_edit.setText(res)
    win._advanced_widgets["detector_threshold"].setValue(0.41)
    # tab 2: a bit of everything
    tab.audio_row.edit.setText(audio); tab.labels_row.edit.setText(res)
    tab.output_row.edit.setText(str(tmp_path / "specout"))
    tab.freq_min_spin.setValue(600); tab.freq_max_spin.setValue(3000)
    tab.formant_ceiling_spin.setValue(9000); tab.padding_spin.setValue(0.5)
    tab.fft_combo.setCurrentText("1024"); tab.color_combo.setCurrentText("inferno")
    tab.show_f2_check.setChecked(False); tab.show_boxes_check.setChecked(False); tab.extend_freq_check.setChecked(False)
    tab.f0_color_btn.set_hex("#123456"); tab.element_color_btns["Wail"].set_hex("#abcdef")
    tab.export_width_spin.setValue(10); tab.export_height_spin.setValue(4); tab.dpi_spin.setValue(200)
    tab.batch_png_check.setChecked(False); tab.batch_workers_spin.setValue(3)

    cfg_path = str(tmp_path / "all.cfg")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (cfg_path, "")))
    win._save_config()
    raw = configparser.ConfigParser(); raw.read(cfg_path, encoding="utf-8")
    assert set(raw.sections()) == {"hyraxid", "spectrogram"}
    assert raw["spectrogram"]["recordings_dir"] == audio and raw["spectrogram"]["freq_min"] == "600"
    assert raw["hyraxid"]["detector_threshold"] == "0.41"

    # restore into a fresh window with default values
    import main_window as MW
    fresh = MW.MainWindow()
    fresh._shown = []
    try:
        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (cfg_path, "")))
        fresh._load_config()
        t = fresh.spectrogram_tab
        assert fresh._advanced_widgets["detector_threshold"].value() == 0.41           # tab 1
        assert (t.freq_min_spin.value(), t.freq_max_spin.value()) == (600, 3000)       # interdependent limits
        assert t.formant_ceiling_spin.value() == 9000 and t.padding_spin.value() == 0.5
        assert t.fft_combo.currentText() == "1024" and t.color_combo.currentText() == "inferno"
        assert not t.show_f2_check.isChecked() and not t.show_boxes_check.isChecked() and not t.extend_freq_check.isChecked()
        assert t.show_f0_check.isChecked()
        assert t.f0_color_btn.hex() == "#123456" and t.element_color_btns["Wail"].hex() == "#abcdef"
        assert t.element_color_btns["Chuck"].hex() == "#1e90ff"                          # untouched default
        assert (t.export_width_spin.value(), t.export_height_spin.value(), t.dpi_spin.value()) == (10, 4, 200)
        assert not t.batch_png_check.isChecked() and t.batch_rec_csv_check.isChecked() and t.batch_workers_spin.value() == 3
        assert t.audio_row.text() == audio and t.output_row.text() == str(tmp_path / "specout")
        assert t.animal_combo.count() == 3                                               # library reloaded
        assert "Settings loaded" in fresh.status_label.text()
    finally:
        fresh.spectrogram_tab.shutdown(); fresh.close(); fresh.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_old_settings_files_without_the_viewer_section_still_load(win, tmp_path, monkeypatch):
    tab = win.spectrogram_tab
    tab.freq_min_spin.setValue(700)
    old = tmp_path / "old.cfg"
    old.write_text("[hyraxid]\ndetector_threshold = 0.4\n", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(old), "")))
    win._load_config()
    assert win._advanced_widgets["detector_threshold"].value() == 0.4
    assert tab.freq_min_spin.value() == 700                  # viewer settings stay as they were


def test_garbage_values_in_the_viewer_section_are_ignored(win, tmp_path, monkeypatch):
    tab = win.spectrogram_tab
    f = tmp_path / "bad.cfg"
    f.write_text("[hyraxid]\n[spectrogram]\nfreq_min = abc\nfft_size = 777\ncolor_f0 = notacolor\nshow_f1 = maybe\n"
                 "recordings_dir = /no/such\nresults_dir = /nor/this\n", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(f), "")))
    win._load_config()
    assert tab.freq_min_spin.value() == 200 and tab.fft_combo.currentText() == "2048"
    assert tab.f0_color_btn.hex() == "#00ff00" and not tab.show_f1_check.isChecked()
    assert "does not exist" in tab.status_label.text() and win._shown == []          # no popup while loading


# ------------------------------------------------------------------------------ optional dependencies
def test_without_the_viewer_packages_the_app_still_works(qapp, monkeypatch):
    import spectrogram_tab as ST
    monkeypatch.setattr(ST, "_IMPORT_ERROR", "ModuleNotFoundError: No module named 'parselmouth'")
    tab = ST.SpectrogramTab()
    assert not tab.available and tab.get_settings() == {}
    tab.set_settings({"freq_min": "700"})              # harmless
    tab.shutdown(); tab.notify_detection_finished()
    from PySide6.QtWidgets import QLabel
    texts = [label.text() for label in tab.findChildren(QLabel)]
    assert any("pip install praat-parselmouth" in t and "parselmouth" in t for t in texts)


# ------------------------------------------------------------------------------ entry point
def test_main_module_is_light_to_import():
    """Spawned batch workers re-import app/main.py; it must not pull in Qt, the GUI or torch."""
    code = ("import sys; sys.path.insert(0, %r); import main; "
            "bad = [m for m in ('PySide6', 'main_window', 'torch', 'ultralytics') if m in sys.modules]; "
            "print('HEAVY:' + ','.join(bad))") % os.path.join(ROOT, "app")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "NUMBA_CACHE_DIR": os.environ.get("TMPDIR", "/tmp")})
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip().endswith("HEAVY:"), out.stdout
