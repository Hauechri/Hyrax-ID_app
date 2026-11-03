"""
Desktop prototype for the Acoustic Spectrogram Viewer.

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
    QColorDialog, QScrollArea,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectrogram_core import SpectrogramProject, RenderSettings, BoutNotFoundError, DEFAULT_ELEMENT_COLOR_MAP

if "__compiled__" in globals():
    # Nuitka standalone build: the exe sits directly next to assets/ (see build_exe.ps1);
    # data paths default to that folder too.
    APP_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
    ASSETS_DIR = os.path.join(APP_DIR, "assets")
else:
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

ICON_PATH = os.path.join(ASSETS_DIR, "icon.ico")


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Acoustic Spectrogram Viewer")
        if os.path.isfile(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
        self.resize(1400, 850)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.project = None
        self.annotation_path = os.path.join(APP_DIR, "Songs analysed by Héloïse.xlsx")
        self.audio_dir_path = os.path.join(APP_DIR, "INPUT")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_canvas())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        if os.path.isfile(self.annotation_path):
            self._load_project()
        else:
            self.status.showMessage("Choose an annotation file and audio folder, then click Load library.")

    # ---------- UI construction ----------

    def _build_controls(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # --- data source ---
        source_box = QGroupBox("Data source")
        sform = QVBoxLayout(source_box)

        self.annotation_edit = QLineEdit()
        self.annotation_edit.setReadOnly(True)
        self._set_path_display(self.annotation_edit, self.annotation_path)
        ann_row = QHBoxLayout()
        ann_row.addWidget(QLabel("Annotation file:"))
        browse_ann_btn = QPushButton("Browse…")
        browse_ann_btn.clicked.connect(self._browse_annotation_file)
        ann_row.addWidget(browse_ann_btn)
        sform.addLayout(ann_row)
        sform.addWidget(self.annotation_edit)

        self.audio_edit = QLineEdit()
        self.audio_edit.setReadOnly(True)
        self._set_path_display(self.audio_edit, self.audio_dir_path)
        audio_row = QHBoxLayout()
        audio_row.addWidget(QLabel("Audio folder:"))
        browse_audio_btn = QPushButton("Browse…")
        browse_audio_btn.clicked.connect(self._browse_audio_folder)
        audio_row.addWidget(browse_audio_btn)
        sform.addLayout(audio_row)
        sform.addWidget(self.audio_edit)

        self.load_btn = QPushButton("Load library")
        self.load_btn.clicked.connect(self._load_project)
        sform.addWidget(self.load_btn)
        layout.addWidget(source_box)

        # --- recording selection ---
        selection_box = QGroupBox("Recording")
        form = QFormLayout(selection_box)
        self.animal_combo = QComboBox()
        self.recording_combo = QComboBox()
        self.bout_combo = QComboBox()
        self.animal_combo.currentTextChanged.connect(self._on_animal_changed)
        self.recording_combo.currentTextChanged.connect(self._on_recording_changed)
        form.addRow("Animal", self.animal_combo)
        form.addRow("Recording", self.recording_combo)
        form.addRow("Bout", self.bout_combo)
        layout.addWidget(selection_box)

        # --- settings ---
        settings_box = QGroupBox("Settings")
        sform2 = QFormLayout(settings_box)

        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0, 10)
        self.padding_spin.setValue(1.0)
        sform2.addRow("Padding (s)", self.padding_spin)

        self.freq_min_spin = QSpinBox()
        self.freq_min_spin.setRange(0, 20000)
        self.freq_min_spin.setValue(200)
        sform2.addRow("Freq min (Hz)", self.freq_min_spin)

        self.freq_max_spin = QSpinBox()
        self.freq_max_spin.setRange(100, 40000)
        self.freq_max_spin.setValue(5000)
        sform2.addRow("Freq max (Hz)", self.freq_max_spin)

        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["512", "1024", "2048", "4096"])
        self.fft_combo.setCurrentText("2048")
        sform2.addRow("FFT size", self.fft_combo)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["grayscale", "classic", "inferno"])
        sform2.addRow("Color scheme", self.color_combo)

        self.formant_ceiling_spin = QSpinBox()
        self.formant_ceiling_spin.setRange(3000, 20000)
        self.formant_ceiling_spin.setSingleStep(500)
        self.formant_ceiling_spin.setValue(7500)
        self.formant_ceiling_spin.setToolTip("Praat formant ceiling for the F1–F3 values in the CSV export.")
        sform2.addRow("Formant ceiling (Hz)", self.formant_ceiling_spin)

        layout.addWidget(settings_box)

        # --- legend colors ---
        legend_box = QGroupBox("Legend colors")
        lform = QFormLayout(legend_box)

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

        self.hz_source_combo = QComboBox()
        self.hz_source_combo.addItem("Detected automatically", "auto")
        self.hz_source_combo.addItem("Annotation file", "annotation")
        self.hz_source_combo.currentIndexChanged.connect(
            lambda _: self.show_annotation_check.setEnabled(self.hz_source_combo.currentData() == "auto"))
        lform.addRow("HzMin / HzMax from", self.hz_source_combo)

        self.show_annotation_check = QCheckBox("Outline annotated boxes (compare)")
        self.show_annotation_check.setChecked(True)
        lform.addRow(self.show_annotation_check)

        self.element_color_btns = {}
        for elem, default_hex in DEFAULT_ELEMENT_COLOR_MAP.items():
            btn = ColorButton(default_hex)
            self.element_color_btns[elem] = btn
            lform.addRow(QLabel(elem), btn)

        layout.addWidget(legend_box)

        # --- export ---
        export_box = QGroupBox("Export")
        eform = QFormLayout(export_box)
        self.export_width_spin = QDoubleSpinBox()
        self.export_width_spin.setRange(2, 40)
        self.export_width_spin.setValue(14)
        eform.addRow("Width (in)", self.export_width_spin)
        self.export_height_spin = QDoubleSpinBox()
        self.export_height_spin.setRange(2, 40)
        self.export_height_spin.setValue(6)
        eform.addRow("Height (in)", self.export_height_spin)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(150)
        eform.addRow("DPI", self.dpi_spin)
        layout.addWidget(export_box)

        self.generate_btn = QPushButton("Generate spectrogram")
        self.generate_btn.clicked.connect(self._on_generate)
        layout.addWidget(self.generate_btn)

        self.save_btn = QPushButton("Save image…")
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        self.export_csv_btn = QPushButton("Export F0 + formants (CSV)…")
        self.export_csv_btn.clicked.connect(self._on_export_csv)
        self.export_csv_btn.setEnabled(False)
        layout.addWidget(self.export_csv_btn)

        layout.addStretch(1)

        # Scrollable wrapper: if the window's usable height is shorter than
        # all the group boxes stacked at their natural size (e.g. maximized
        # on a shorter display), scroll instead of squishing every row.
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(480)
        return scroll

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

    # ---------- data loading ----------

    @staticmethod
    def _set_path_display(line_edit, path):
        # Show just the last component — the full path only fits by scrolling
        # to whichever end the cursor happens to be at, which reads as garbled.
        line_edit.setText(os.path.basename(os.path.normpath(path)) or path)
        line_edit.setToolTip(path)

    def _browse_annotation_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose annotation file", self.annotation_path, "Excel files (*.xlsx *.xls)")
        if path:
            self.annotation_path = path
            self._set_path_display(self.annotation_edit, path)

    def _browse_audio_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose audio library folder", self.audio_dir_path)
        if path:
            self.audio_dir_path = path
            self._set_path_display(self.audio_edit, path)

    def _load_project(self):
        try:
            self.project = SpectrogramProject(
                APP_DIR,
                annotation_file=self.annotation_path,
                audio_base_dir=self.audio_dir_path,
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
        self._on_animal_changed(self.animal_combo.currentText())

        if not singers:
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
        else:
            self.status.showMessage(f"Loaded {len(self.project.ann)} annotation rows, {len(singers)} animal(s) found.")

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
            bouts = self.project.list_bouts(animal, song_id)
            self.bout_combo.addItems([str(b) for b in bouts])

    # ---------- actions ----------

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
            self.save_btn.setEnabled(True)
            self.last_measurements = (animal, song_id, int(bout_text), meta["measurements"])
            self.export_csv_btn.setEnabled(True)
            msg = f"Bout {bout_text} spans {meta['t_start']:.2f}s – {meta['t_end']:.2f}s."
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
        animal = self.animal_combo.currentText()
        song_id = self.recording_combo.currentText()
        bout_text = self.bout_combo.currentText()
        default_path = self.project.default_output_path(animal, song_id, int(bout_text))
        path, _ = QFileDialog.getSaveFileName(self, "Save spectrogram image", default_path, "PNG image (*.png)")
        if not path:
            return
        settings = self._current_settings()
        try:
            self.project.save_figure(
                self.fig, path, settings.image_dpi,
                width_in=settings.figure_width, height_in=settings.figure_height,
            )
            self.status.showMessage(f"Saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_export_csv(self):
        animal, song_id, bout, measurements = self.last_measurements
        default_path = os.path.splitext(self.project.default_output_path(animal, song_id, bout))[0] + "_hz.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export F0 + formants", default_path, "CSV file (*.csv)")
        if not path:
            return
        try:
            self.project.export_measurements(measurements, path, animal, song_id, bout)
            self.status.showMessage(f"Exported {len(measurements)} element(s) to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))


def main():
    app = QApplication(sys.argv)
    if os.path.isfile(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
