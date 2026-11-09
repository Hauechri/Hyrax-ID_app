"""
Desktop app for the Acoustic Spectrogram Viewer.

Run with:
    .venv\\Scripts\\python.exe app\\main.py
"""
import os
import statistics
import sys
import traceback

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QMessageBox, QStatusBar, QSplitter, QLineEdit,
    QColorDialog, QScrollArea, QTabWidget, QProgressBar, QPlainTextEdit,
)
from PySide6.QtCore import Qt, QObject, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QIcon, QDesktopServices
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectrogram_core import SpectrogramProject, RenderSettings, BoutNotFoundError, DEFAULT_ELEMENT_COLOR_MAP
from call_detection import BOUT_GAP_S

if "__compiled__" in globals():
    # Nuitka standalone build: the exe sits directly next to assets/ (see build_exe.ps1);
    # data paths default to that folder too.
    APP_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
    ASSETS_DIR = os.path.join(APP_DIR, "assets")
else:
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

ICON_PATH = os.path.join(ASSETS_DIR, "icon.ico")
ALL_ANIMALS = "All animals"


class ColorButton(QPushButton):
    """A small swatch button that opens a color picker and remembers a hex color."""

    def __init__(self, initial_hex, parent=None):
        super().__init__(parent)
        self._hex = initial_hex
        self.setFixedSize(32, 24)
        self.clicked.connect(self._pick)
        self._refresh()

    def _refresh(self):
        self.setToolTip(self._hex)
        self.setStyleSheet(f"background-color: {self._hex}; border: 1px solid #888; border-radius: 3px;")

    def _pick(self):
        # Parent to the top-level window (not `self`) and clear any inherited
        # stylesheet — otherwise this button's flat background-color style
        # cascades down into the dialog and paints the whole picker solid.
        dlg = QColorDialog(QColor(self._hex), self.window())
        dlg.setStyleSheet("")
        if dlg.exec() == QColorDialog.DialogCode.Accepted:
            color = dlg.selectedColor()
            if color.isValid():
                self._hex = color.name()
                self._refresh()

    def hex(self):
        return self._hex


class PathRow(QWidget):
    """Read-only path field (shows the last path component, full path as tooltip) + Browse button."""

    def __init__(self, path, browse, extra_button=None):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        self.button = QPushButton("Browse…")
        self.button.clicked.connect(browse)
        row.addWidget(self.edit, 1)
        row.addWidget(self.button)
        if extra_button is not None:
            row.addWidget(extra_button)
        self.set_path(path)

    def set_path(self, path):
        # Show just the last component — the full path only fits by scrolling
        # to whichever end the cursor happens to be at, which reads as garbled.
        self.edit.setText(os.path.basename(os.path.normpath(path)) or path)
        self.edit.setToolTip(path)


class BatchWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, project, recordings, settings, outputs):
        super().__init__()
        self.project, self.recordings, self.settings, self.outputs = project, recordings, settings, outputs
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            result = self.project.run_batch(
                self.recordings, self.settings, **self.outputs,
                progress=lambda done, total, msg: self.progress.emit(done, total, msg),
                should_stop=lambda: self._stop,
            )
            result["cancelled"] = self._stop
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Acoustic Spectrogram Viewer")
        if os.path.isfile(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
        self.resize(1400, 900)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.project = None
        self.last_render = None
        self._batch_thread = None
        self._batch_worker = None
        self.annotation_path = os.path.join(APP_DIR, "Songs analysed by Héloïse.xlsx")
        self.audio_dir_path = os.path.join(APP_DIR, "INPUT")
        self.output_dir_path = os.path.join(APP_DIR, "OUTPUT")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_canvas())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.use_annotation_check.setChecked(os.path.isfile(self.annotation_path))
        self._on_annotation_toggled(self.use_annotation_check.isChecked())
        if os.path.isdir(self.audio_dir_path):
            self._load_project()
        else:
            self.status.showMessage("Choose an audio folder, then click Load library.")

    # ---------- UI construction ----------

    def _build_controls(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self._build_data_box())
        layout.addWidget(self._build_output_box())

        self.mode_tabs = QTabWidget()
        self.mode_tabs.addTab(self._build_single_tab(), "Single recording")
        self.mode_tabs.addTab(self._build_batch_tab(), "Batch (folder)")
        layout.addWidget(self.mode_tabs)

        layout.addWidget(self._build_measurement_box())
        layout.addWidget(self._build_display_box())
        layout.addWidget(self._build_legend_box())
        layout.addWidget(self._build_image_box())
        layout.addStretch(1)

        # Scrollable wrapper: if the window's usable height is shorter than
        # all the group boxes stacked at their natural size (e.g. maximized
        # on a shorter display), scroll instead of squishing every row.
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(540)
        return scroll

    def _build_data_box(self):
        self.data_box = QGroupBox("1 · Data")
        v = QVBoxLayout(self.data_box)

        audio_label = QLabel("Audio folder (.wav files, any sub-folder layout):")
        audio_label.setWordWrap(True)
        v.addWidget(audio_label)
        self.audio_row = PathRow(self.audio_dir_path, self._browse_audio_folder)
        v.addWidget(self.audio_row)

        self.use_annotation_check = QCheckBox("Use annotation file")
        self.use_annotation_check.setToolTip(
            "On: bouts, element times and reference HzMin / HzMax come from the Excel file.\n"
            "Off: calls are detected automatically from the audio.")
        self.use_annotation_check.toggled.connect(self._on_annotation_toggled)
        v.addWidget(self.use_annotation_check)

        self.annotation_row = PathRow(self.annotation_path, self._browse_annotation_file)
        v.addWidget(self.annotation_row)

        self.annotation_hint = QLabel()
        self.annotation_hint.setWordWrap(True)
        self.annotation_hint.setStyleSheet("color: gray;")
        v.addWidget(self.annotation_hint)

        self.load_btn = QPushButton("Load library")
        self.load_btn.clicked.connect(self._load_project)
        v.addWidget(self.load_btn)
        return self.data_box

    def _build_output_box(self):
        box = QGroupBox("2 · Output folder (images + CSV)")
        v = QVBoxLayout(box)
        open_btn = QPushButton("Open")
        open_btn.setToolTip("Open the output folder in Explorer")
        open_btn.clicked.connect(self._open_output_folder)
        self.output_row = PathRow(self.output_dir_path, self._browse_output_folder, extra_button=open_btn)
        v.addWidget(self.output_row)
        hint = QLabel("Files go to <output>/<Animal>/<Recording>/, e.g. MS_240701_0091_bout_01.png")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        v.addWidget(hint)
        return box

    def _build_single_tab(self):
        tab = QWidget()
        v = QVBoxLayout(tab)
        form = QFormLayout()
        self.animal_combo = QComboBox()
        self.recording_combo = QComboBox()
        self.bout_combo = QComboBox()
        self.animal_combo.currentTextChanged.connect(self._on_animal_changed)
        self.recording_combo.currentTextChanged.connect(self._on_recording_changed)
        form.addRow("Animal", self.animal_combo)
        form.addRow("Recording", self.recording_combo)
        form.addRow("Bout", self.bout_combo)
        v.addLayout(form)

        self.generate_btn = QPushButton("Generate spectrogram")
        self.generate_btn.clicked.connect(self._on_generate)
        v.addWidget(self.generate_btn)

        row = QHBoxLayout()
        self.save_btn = QPushButton("Save image")
        self.save_btn.clicked.connect(self._on_save)
        self.export_csv_btn = QPushButton("Save CSV")
        self.export_csv_btn.clicked.connect(self._on_export_csv)
        for b in (self.save_btn, self.export_csv_btn):
            b.setEnabled(False)
            row.addWidget(b)
        v.addLayout(row)
        return tab

    def _build_batch_tab(self):
        tab = QWidget()
        v = QVBoxLayout(tab)
        form = QFormLayout()
        self.batch_animal_combo = QComboBox()
        self.batch_animal_combo.currentTextChanged.connect(self._update_batch_summary)
        form.addRow("Animal", self.batch_animal_combo)
        v.addLayout(form)

        self.batch_summary = QLabel("Load a library first.")
        self.batch_summary.setStyleSheet("color: gray;")
        v.addWidget(self.batch_summary)

        self.batch_png_check = QCheckBox("PNG image per bout")
        self.batch_rec_csv_check = QCheckBox("CSV per recording")
        self.batch_all_csv_check = QCheckBox("One combined CSV for the whole batch")
        for c in (self.batch_png_check, self.batch_rec_csv_check, self.batch_all_csv_check):
            c.setChecked(True)
            v.addWidget(c)

        row = QHBoxLayout()
        self.batch_run_btn = QPushButton("Run batch")
        self.batch_run_btn.clicked.connect(self._on_run_batch)
        self.batch_cancel_btn = QPushButton("Cancel")
        self.batch_cancel_btn.setEnabled(False)
        self.batch_cancel_btn.clicked.connect(self._on_cancel_batch)
        row.addWidget(self.batch_run_btn)
        row.addWidget(self.batch_cancel_btn)
        v.addLayout(row)

        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        v.addWidget(self.batch_progress)
        self.batch_log = QPlainTextEdit()
        self.batch_log.setReadOnly(True)
        self.batch_log.setMaximumHeight(130)
        v.addWidget(self.batch_log)
        return tab

    def _build_measurement_box(self):
        box = QGroupBox("Measurement")
        form = QFormLayout(box)

        self.hz_source_combo = QComboBox()
        self.hz_source_combo.addItem("Detected automatically", "auto")
        self.hz_source_combo.addItem("Annotation file", "annotation")
        self.hz_source_combo.currentIndexChanged.connect(self._update_measurement_controls)
        form.addRow("HzMin / HzMax boxes", self.hz_source_combo)

        self.show_annotation_check = QCheckBox("Outline annotated boxes (compare)")
        self.show_annotation_check.setChecked(True)
        form.addRow(self.show_annotation_check)

        self.formant_ceiling_spin = QSpinBox()
        self.formant_ceiling_spin.setRange(3000, 20000)
        self.formant_ceiling_spin.setSingleStep(500)
        self.formant_ceiling_spin.setValue(7500)
        self.formant_ceiling_spin.setSuffix(" Hz")
        self.formant_ceiling_spin.setToolTip("Praat formant ceiling for F1–F3 (lines and CSV).")
        form.addRow("Formant ceiling", self.formant_ceiling_spin)
        return box

    def _build_display_box(self):
        box = QGroupBox("Display")
        form = QFormLayout(box)

        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0, 10)
        self.padding_spin.setValue(1.0)
        self.padding_spin.setSuffix(" s")
        form.addRow("Padding", self.padding_spin)

        self.freq_min_spin = QSpinBox()
        self.freq_min_spin.setRange(0, 20000)
        self.freq_min_spin.setValue(200)
        self.freq_min_spin.setSuffix(" Hz")
        form.addRow("Freq min", self.freq_min_spin)

        self.freq_max_spin = QSpinBox()
        self.freq_max_spin.setRange(100, 40000)
        self.freq_max_spin.setValue(5000)
        self.freq_max_spin.setSuffix(" Hz")
        form.addRow("Freq max", self.freq_max_spin)

        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["512", "1024", "2048", "4096"])
        self.fft_combo.setCurrentText("2048")
        form.addRow("FFT size", self.fft_combo)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["grayscale", "classic", "inferno"])
        form.addRow("Color scheme", self.color_combo)
        return box

    def _build_legend_box(self):
        box = QGroupBox("Lines && colors")
        lform = QFormLayout(box)

        self.show_f0_check = QCheckBox("F0 (pitch)")
        self.show_f0_check.setChecked(True)
        self.f0_color_btn = ColorButton("#00FF00")
        lform.addRow(self.show_f0_check, self.f0_color_btn)

        self.show_f1_check = QCheckBox("F1 (1st formant)")
        self.show_f1_check.setChecked(True)
        self.f1_color_btn = ColorButton("#00FFFF")
        lform.addRow(self.show_f1_check, self.f1_color_btn)

        self.show_f2_check = QCheckBox("F2 (2nd formant)")
        self.show_f2_check.setChecked(True)
        self.f2_color_btn = ColorButton("#FFD700")
        lform.addRow(self.show_f2_check, self.f2_color_btn)

        self.show_boxes_check = QCheckBox("HzMin / HzMax boxes")
        self.show_boxes_check.setChecked(True)
        lform.addRow(self.show_boxes_check)

        self.element_color_btns = {}
        for elem, default_hex in DEFAULT_ELEMENT_COLOR_MAP.items():
            btn = ColorButton(default_hex)
            self.element_color_btns[elem] = btn
            label = "Call (detected)" if elem == "Call" else elem
            lform.addRow(QLabel(label), btn)
        return box

    def _build_image_box(self):
        box = QGroupBox("Image size")
        eform = QFormLayout(box)
        self.export_width_spin = QDoubleSpinBox()
        self.export_width_spin.setRange(2, 40)
        self.export_width_spin.setValue(14)
        self.export_width_spin.setSuffix(" in")
        eform.addRow("Width", self.export_width_spin)
        self.export_height_spin = QDoubleSpinBox()
        self.export_height_spin.setRange(2, 40)
        self.export_height_spin.setValue(6)
        self.export_height_spin.setSuffix(" in")
        eform.addRow("Height", self.export_height_spin)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(150)
        eform.addRow("DPI", self.dpi_spin)
        return box

    def _build_canvas(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # One persistent figure/canvas, reused (cleared + redrawn) on every
        # Generate call, so the on-screen size always matches the widget
        # instead of jumping to a fixed size each time.
        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, container)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        return container

    # ---------- data source ----------

    def _on_annotation_toggled(self, checked):
        self.annotation_row.setEnabled(checked)
        self.annotation_hint.setText(
            "Bouts, element times and the reference HzMin / HzMax come from the annotation file."
            if checked else
            f"Calls are detected automatically from the audio and grouped into bouts by pauses "
            f"longer than {BOUT_GAP_S:g} s. Element type is not classified.")
        self._update_measurement_controls()
        if self.project is not None and self.project.uses_annotation != checked:
            self.status.showMessage("Click Load library to apply the annotation setting.")

    def _update_measurement_controls(self):
        has_ann = self.use_annotation_check.isChecked()
        ann_item = self.hz_source_combo.model().item(1)
        ann_item.setEnabled(has_ann)
        if not has_ann and self.hz_source_combo.currentData() == "annotation":
            self.hz_source_combo.setCurrentIndex(0)
        self.show_annotation_check.setEnabled(has_ann and self.hz_source_combo.currentData() == "auto")

    def _browse_annotation_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose annotation file", self.annotation_path, "Excel files (*.xlsx *.xls)")
        if path:
            self.annotation_path = path
            self.annotation_row.set_path(path)

    def _browse_audio_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose audio library folder", self.audio_dir_path)
        if path:
            self.audio_dir_path = path
            self.audio_row.set_path(path)

    def _browse_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_dir_path)
        if path:
            self.output_dir_path = path
            self.output_row.set_path(path)
            if self.project is not None:
                self.project.output_base_dir = path

    def _open_output_folder(self):
        os.makedirs(self.output_dir_path, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_dir_path))

    def _load_project(self):
        use_ann = self.use_annotation_check.isChecked()
        try:
            self.status.showMessage("Loading library…")
            QApplication.processEvents()
            self.project = SpectrogramProject(
                self.audio_dir_path,
                annotation_file=self.annotation_path if use_ann else None,
                output_base_dir=self.output_dir_path,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load library",
                                 f"Could not load the annotation file / audio folder:\n{exc}")
            return

        singers = self.project.list_singers()
        self.animal_combo.blockSignals(True)
        self.animal_combo.clear()
        self.animal_combo.addItems(singers)
        self.animal_combo.blockSignals(False)
        self.batch_animal_combo.blockSignals(True)
        self.batch_animal_combo.clear()
        self.batch_animal_combo.addItems([ALL_ANIMALS] + singers)
        self.batch_animal_combo.blockSignals(False)
        self._on_animal_changed(self.animal_combo.currentText())
        self._update_batch_summary()

        n_wav = len(self.project._wav_index)
        if not singers and use_ann:
            raw_singers = sorted(self.project.ann["Singer"].dropna().astype(str).unique())
            QMessageBox.warning(
                self, "No matching audio found",
                f"The annotation file has {len(self.project.ann)} rows listing these animals: "
                f"{raw_singers}\n\n"
                f"but no .wav file anywhere under:\n{self.audio_dir_path}\n\n"
                f"matches any of the recording IDs in the annotation file. The audio folder can "
                f"have any layout (subfolders per animal, or all files in one folder) — it just "
                f"needs to contain files named <Recording ID>.wav."
            )
            self.status.showMessage("Loaded annotation file, but found no matching audio files.")
        elif not singers:
            self.status.showMessage(f"No .wav files found under {self.audio_dir_path}.")
        elif use_ann:
            self.status.showMessage(
                f"Loaded {len(self.project.ann)} annotation rows, {n_wav} audio file(s), {len(singers)} animal(s).")
        else:
            self.status.showMessage(
                f"Loaded {n_wav} audio file(s), {len(singers)} animal(s) — calls will be detected automatically.")

    def _on_animal_changed(self, animal):
        self.recording_combo.blockSignals(True)
        self.recording_combo.clear()
        if animal and self.project:
            self.recording_combo.addItems([str(s) for s in self.project.list_recordings(animal)])
        self.recording_combo.blockSignals(False)
        self._on_recording_changed(self.recording_combo.currentText())

    def _on_recording_changed(self, song_id):
        self.bout_combo.clear()
        animal = self.animal_combo.currentText()
        if animal and song_id and self.project:
            if not self.project.uses_annotation:
                self.status.showMessage(f"Detecting calls in {song_id}…")
                QApplication.processEvents()
            try:
                bouts = self.project.list_bouts(animal, song_id)
            except Exception as exc:
                QMessageBox.warning(self, "Could not read recording", str(exc))
                return
            self.bout_combo.addItems([str(b) for b in bouts])
            if not self.project.uses_annotation:
                self.status.showMessage(f"{song_id}: {len(bouts)} bout(s) detected.")

    # ---------- single recording ----------

    def _current_settings(self):
        return RenderSettings(
            padding_seconds=self.padding_spin.value(),
            freq_min_hz=self.freq_min_spin.value(),
            freq_max_hz=self.freq_max_spin.value(),
            show_F0=self.show_f0_check.isChecked(),
            show_F1=self.show_f1_check.isChecked(),
            show_F2=self.show_f2_check.isChecked(),
            show_hz_boxes=self.show_boxes_check.isChecked(),
            hz_source=self.hz_source_combo.currentData(),
            show_annotation_overlay=self.show_annotation_check.isChecked(),
            formant_ceiling_hz=self.formant_ceiling_spin.value(),
            color_scheme=self.color_combo.currentText(),
            fft_size=int(self.fft_combo.currentText()),
            color_f0=self.f0_color_btn.hex(),
            color_f1=self.f1_color_btn.hex(),
            color_f2=self.f2_color_btn.hex(),
            element_colors={elem: btn.hex() for elem, btn in self.element_color_btns.items()},
            figure_width=self.export_width_spin.value(),
            figure_height=self.export_height_spin.value(),
            image_dpi=self.dpi_spin.value(),
        )

    def _on_generate(self):
        animal = self.animal_combo.currentText()
        song_id = self.recording_combo.currentText()
        bout_text = self.bout_combo.currentText()
        if not (animal and song_id and bout_text):
            QMessageBox.warning(self, "Nothing selected", "Choose an animal, recording and bout first.")
            return

        self.generate_btn.setEnabled(False)
        self.status.showMessage("Rendering…")
        QApplication.processEvents()
        try:
            _, meta = self.project.render(animal, song_id, int(bout_text), self._current_settings(), fig=self.fig)
            self.canvas.draw()
            self.last_render = (animal, song_id, int(bout_text), meta["measurements"])
            self.save_btn.setEnabled(True)
            self.export_csv_btn.setEnabled(True)
            n = len(meta["measurements"])
            msg = f"Bout {bout_text}: {n} element(s), {meta['t_start']:.2f}s – {meta['t_end']:.2f}s."
            diffs = [abs(m[f"auto_hz{k}"] - m[f"annotated_hz{k}"])
                     for m in meta["measurements"] for k in ("max", "min")
                     if m[f"auto_hz{k}"] == m[f"auto_hz{k}"] and m[f"annotated_hz{k}"] == m[f"annotated_hz{k}"]]
            if diffs:
                msg += (f"  Detected vs annotated: median |diff| {statistics.median(diffs):.0f} Hz"
                        f" over {len(diffs) // 2} element(s).")
            self.status.showMessage(msg)
        except BoutNotFoundError as exc:
            QMessageBox.warning(self, "No data", str(exc))
            self.status.showMessage("No data for that selection.")
        except Exception:
            QMessageBox.critical(self, "Render failed", traceback.format_exc())
            self.status.showMessage("Render failed.")
        finally:
            self.generate_btn.setEnabled(True)

    def _on_save(self):
        animal, song_id, bout, _ = self.last_render
        path = self.project.bout_path(animal, song_id, bout, ".png")
        settings = self._current_settings()
        try:
            self.project.save_figure(
                self.fig, path, settings.image_dpi,
                width_in=settings.figure_width, height_in=settings.figure_height,
            )
            self.status.showMessage(f"Saved image to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_export_csv(self):
        animal, song_id, bout, measurements = self.last_render
        path = self.project.bout_path(animal, song_id, bout, "_hz.csv")
        try:
            self.project.export_measurements(measurements, path, animal, song_id, bout)
            self.status.showMessage(f"Saved {len(measurements)} element(s) to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ---------- batch ----------

    def _batch_recordings(self):
        if not self.project:
            return []
        choice = self.batch_animal_combo.currentText()
        animals = self.project.list_singers() if choice == ALL_ANIMALS else [choice]
        return [(a, s) for a in animals if a for s in self.project.list_recordings(a)]

    def _update_batch_summary(self, *_):
        recs = self._batch_recordings()
        if not self.project:
            self.batch_summary.setText("Load a library first.")
            return
        how = "bouts from the annotation file" if self.project.uses_annotation else "calls detected automatically"
        self.batch_summary.setText(f"{len(recs)} recording(s) — {how}.")

    def _on_run_batch(self):
        recordings = self._batch_recordings()
        if not recordings:
            QMessageBox.warning(self, "Nothing to process", "No recordings found for this selection.")
            return
        outputs = dict(save_png=self.batch_png_check.isChecked(),
                       csv_per_recording=self.batch_rec_csv_check.isChecked(),
                       combined_csv=self.batch_all_csv_check.isChecked())
        if not any(outputs.values()):
            QMessageBox.warning(self, "No output selected", "Tick at least one output.")
            return

        self.project.output_base_dir = self.output_dir_path
        self.batch_log.clear()
        self.batch_log.appendPlainText(f"Processing {len(recordings)} recording(s) → {self.output_dir_path}")
        self.batch_progress.setRange(0, 0)  # busy until the bout count is known
        self._set_batch_running(True)

        self._batch_thread = QThread(self)
        self._batch_worker = BatchWorker(self.project, recordings, self._current_settings(), outputs)
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_thread.start()

    def _on_cancel_batch(self):
        if self._batch_worker:
            self._batch_worker.stop()
            self.batch_cancel_btn.setEnabled(False)
            self.batch_log.appendPlainText("Cancelling after the current bout…")

    def _on_batch_progress(self, done, total, msg):
        self.batch_progress.setRange(0, total)
        self.batch_progress.setValue(done)
        self.batch_log.appendPlainText(msg)
        self.status.showMessage(f"Batch: {done}/{total} bouts")

    def _on_batch_finished(self, result):
        self._set_batch_running(False)
        state = "Cancelled" if result.get("cancelled") else "Done"
        self.batch_log.appendPlainText(
            f"{state}: {result['bouts']}/{result['total']} bout(s), {len(result['files'])} file(s) written, "
            f"{len(result['errors'])} error(s).")
        for e in result["errors"]:
            self.batch_log.appendPlainText(f"  ! {e}")
        self.status.showMessage(f"Batch {state.lower()}: {len(result['files'])} file(s) in {self.output_dir_path}")

    def _on_batch_failed(self, tb):
        self._set_batch_running(False)
        self.batch_log.appendPlainText("Batch failed.")
        QMessageBox.critical(self, "Batch failed", tb)

    def _set_batch_running(self, running):
        for w in (self.data_box, self.generate_btn, self.save_btn, self.export_csv_btn,
                  self.batch_run_btn, self.batch_animal_combo, self.output_row):
            w.setEnabled(not running)
        if not running:
            self.save_btn.setEnabled(self.last_render is not None)
            self.export_csv_btn.setEnabled(self.last_render is not None)
            self.batch_progress.setRange(0, max(1, self.batch_progress.maximum()))
        self.batch_cancel_btn.setEnabled(running)

    def closeEvent(self, event):
        if self._batch_thread is not None and self._batch_thread.isRunning():
            self._batch_worker.stop()
            self._batch_thread.quit()
            self._batch_thread.wait(5000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    if os.path.isfile(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
