"""
"Spectrogram viewer" tab: shows the bouts found by the detection tab as spectrograms and measures every
element with Praat (F0 range, F0 mean / median, F1-F3).

Port of the Spectrogram-Viewer prototype's window. Instead of an Excel annotation file it reads the Audacity
label files the detection tab writes (`<recording>_labels.txt`) and finds the recordings by name. If the run
had no classification, every bout is listed under the animal "unknown".

The engine lives in engine/hyrax_id/spectro (Qt-free). This module only needs matplotlib and
praat-parselmouth in addition to the detection tab's packages; if they are missing the tab shows what to
install and the rest of the app keeps working.
"""
import os
import traceback

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

    from hyrax_id.spectro.library import (
        DEFAULT_ELEMENT_COLOR_MAP,
        UNKNOWN_ANIMAL,
        BoutNotFoundError,
        RenderSettings,
        ResultsLibrary,
    )

    _IMPORT_ERROR = None
except Exception as _exc:  # ImportError normally; never let the optional viewer break the whole app
    _IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"
    DEFAULT_ELEMENT_COLOR_MAP = {"Wail": "#FF4500", "Chuck": "#1E90FF", "Snort": "#FF00FF"}
    UNKNOWN_ANIMAL = "unknown"

CFG_SECTION = "spectrogram"
ALL_ANIMALS = "All animals"
RERENDER_DELAY_MS = 350
GRAY = "color: gray;"


def _to_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class ColorButton(QPushButton):
    """A small swatch button that opens a color picker and remembers a hex color."""

    changed = Signal(str)

    def __init__(self, initial_hex, parent=None):
        super().__init__(parent)
        self._hex = QColor(initial_hex).name()  # always lower-case '#rrggbb', like a picked colour
        self.setFixedSize(32, 24)
        self.clicked.connect(self._pick)
        self._refresh()

    def _refresh(self):
        self.setToolTip(self._hex)
        self.setStyleSheet(f"background-color: {self._hex}; border: 1px solid #888; border-radius: 3px;")

    def _pick(self):
        # Parent to the top-level window (not `self`) and clear any inherited stylesheet -- otherwise this
        # button's flat background-color style cascades into the dialog and paints the whole picker solid.
        dlg = QColorDialog(QColor(self._hex), self.window())
        dlg.setStyleSheet("")
        if dlg.exec() == QColorDialog.DialogCode.Accepted:
            color = dlg.selectedColor()
            if color.isValid() and color.name() != self._hex:
                self.set_hex(color.name())
                self.changed.emit(self._hex)

    def set_hex(self, value):
        if QColor(value).isValid():
            self._hex = QColor(value).name()
            self._refresh()

    def hex(self):
        return self._hex


class PathRow(QWidget):
    """Editable path field (full path, like the detection tab) + Browse button (+ optional extra button)."""

    def __init__(self, browse, extra_button=None, placeholder=""):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.button = QPushButton("Browse...")
        self.button.clicked.connect(browse)
        row.addWidget(self.edit, 1)
        row.addWidget(self.button)
        if extra_button is not None:
            row.addWidget(extra_button)

    def text(self):
        return self.edit.text().strip()


class BatchWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, library, recordings, settings, outputs, workers):
        super().__init__()
        self.library, self.recordings, self.settings = library, recordings, settings
        self.outputs, self.workers = outputs, workers
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            result = self.library.run_batch(
                self.recordings, self.settings, **self.outputs, workers=self.workers,
                progress=lambda done, total, msg: self.progress.emit(done, total, msg),
                should_stop=lambda: self._stop,
            )
            result["cancelled"] = self._stop
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SpectrogramTab(QWidget):
    def __init__(self, get_detection_paths=None, on_load_settings=None, on_save_settings=None, parent=None):
        """get_detection_paths() -> (recordings folder, results folder) of the detection tab, for the
        'Take from Detection tab' button. on_load_settings / on_save_settings: the window's settings
        handlers (this tab is saved in the same .cfg file as the detection tab)."""
        super().__init__(parent)
        self._get_detection_paths = get_detection_paths
        self._on_load_settings = on_load_settings
        self._on_save_settings = on_save_settings
        self._deps_error = _IMPORT_ERROR
        self.library = None
        self.last_render = None  # (animal, recording, bout, measurements) of the figure on screen
        self._batch_thread = None
        self._batch_worker = None
        self._ready = False  # suppress auto-reload / auto-redraw while building the UI
        self._auto_output = ""  # the output folder we derived ourselves (so a later change may replace it)
        self._loaded_key = None  # (recordings, results) of the library currently loaded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        if self._deps_error:
            msg = QLabel(
                "The spectrogram viewer needs extra packages that are not installed:\n\n"
                f"    {self._deps_error}\n\n"
                "Install them with\n\n    pip install praat-parselmouth matplotlib pandas\n\n"
                "and restart the app. The detection tab works without them."
            )
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setWordWrap(True)
            outer.addWidget(msg)
            return

        self._rerender_timer = QTimer(self)
        self._rerender_timer.setSingleShot(True)
        self._rerender_timer.setInterval(RERENDER_DELAY_MS)
        self._rerender_timer.timeout.connect(lambda: self._on_generate(auto=True))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_canvas())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 720])
        outer.addWidget(splitter)
        self._connect_live_options()
        self._on_recording_changed("")
        self._update_batch_summary()
        self._ready = True

    @property
    def available(self):
        return self._deps_error is None

    # ---------- UI construction ----------

    def _build_controls(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        row = QHBoxLayout()
        for text, handler in (("Load settings...", self._on_load_settings), ("Save settings...", self._on_save_settings)):
            btn = QPushButton(text)
            if handler is not None:
                btn.clicked.connect(handler)
            else:
                btn.setEnabled(False)
            row.addWidget(btn)
        layout.addLayout(row)

        layout.addWidget(self._build_data_box())
        layout.addWidget(self._build_output_box())

        self.mode_tabs = QTabWidget()
        self.single_tab = self._build_single_tab()
        self.mode_tabs.addTab(self.single_tab, "Single recording")
        self.mode_tabs.addTab(self._build_batch_tab(), "Batch (folder)")
        layout.addWidget(self.mode_tabs)

        self.option_boxes = [
            self._build_measurement_box(),
            self._build_display_box(),
            self._build_legend_box(),
            self._build_image_box(),
        ]
        for box in self.option_boxes:
            layout.addWidget(box)
        layout.addStretch(1)

        # Scrollable wrapper: if the usable height is shorter than all the group boxes stacked at their
        # natural size, scroll instead of squishing every row.
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(560)
        return scroll

    def _build_data_box(self):
        self.data_box = QGroupBox("1 · Data")
        v = QVBoxLayout(self.data_box)

        v.addWidget(QLabel("Recordings folder (.wav files, any sub-folder layout):"))
        self.audio_row = PathRow(self._browse_audio_folder)
        self.audio_row.edit.editingFinished.connect(self._on_paths_edited)
        v.addWidget(self.audio_row)

        v.addWidget(QLabel("Results folder (the *_labels.txt files of the Detection tab):"))
        self.labels_row = PathRow(self._browse_labels_folder)
        self.labels_row.edit.editingFinished.connect(self._on_paths_edited)
        v.addWidget(self.labels_row)

        row = QHBoxLayout()
        self.take_btn = QPushButton("Take from Detection tab")
        self.take_btn.setToolTip("Use the input and output folder currently set in the Detection tab.")
        self.take_btn.setEnabled(self._get_detection_paths is not None)
        self.take_btn.clicked.connect(self._take_from_detection)
        self.load_btn = QPushButton("Reload library")
        self.load_btn.setToolTip("Re-read both folders (e.g. after a new detection run).")
        self.load_btn.clicked.connect(lambda: self._load_library(interactive=True))
        row.addWidget(self.take_btn)
        row.addWidget(self.load_btn)
        v.addLayout(row)

        self.data_hint = QLabel("Recordings and label files are matched by name: <name>.wav ↔ <name>_labels.txt.")
        self.data_hint.setWordWrap(True)
        self.data_hint.setStyleSheet(GRAY)
        v.addWidget(self.data_hint)
        return self.data_box

    def _build_output_box(self):
        self.output_box = QGroupBox("2 · Output folder (images + CSV)")
        v = QVBoxLayout(self.output_box)
        open_btn = QPushButton("Open")
        open_btn.setToolTip("Open the output folder in the file explorer")
        open_btn.clicked.connect(self._open_output_folder)
        self.output_row = PathRow(self._browse_output_folder, extra_button=open_btn,
                                  placeholder="default: <results folder>/spectrograms")
        self.output_row.edit.editingFinished.connect(self._on_output_edited)
        v.addWidget(self.output_row)
        hint = QLabel("Files go to <output>/<Animal>/<Recording>/, e.g. FB_240701_0091_bout_01.png")
        hint.setWordWrap(True)
        hint.setStyleSheet(GRAY)
        v.addWidget(hint)
        return self.output_box

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
        self.generate_btn.clicked.connect(lambda: self._on_generate())
        v.addWidget(self.generate_btn)
        live = QLabel("After the first Generate, changing the bout or any option redraws automatically.")
        live.setWordWrap(True)
        live.setStyleSheet(GRAY)
        v.addWidget(live)

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
        self.batch_workers_spin = QSpinBox()
        self.batch_workers_spin.setRange(0, 64)
        self.batch_workers_spin.setSpecialValueText("auto")
        self.batch_workers_spin.setToolTip("Parallel processes. auto = all CPU cores but two (at most 6); 1 = no parallelism.")
        form.addRow("Parallel workers", self.batch_workers_spin)
        v.addLayout(form)

        self.batch_summary = QLabel("Choose the recordings and results folders first.")
        self.batch_summary.setWordWrap(True)
        self.batch_summary.setStyleSheet(GRAY)
        v.addWidget(self.batch_summary)

        self.batch_png_check = QCheckBox("PNG image per bout")
        self.batch_rec_csv_check = QCheckBox("CSV per recording")
        self.batch_all_csv_check = QCheckBox("One combined CSV for the whole batch")
        for c in (self.batch_png_check, self.batch_rec_csv_check, self.batch_all_csv_check):
            c.setChecked(True)
            v.addWidget(c)
        note = QLabel("Uses the Measurement, Display, Lines and Image size options below, "
                      "as they are when you click Run batch.")
        note.setWordWrap(True)
        note.setStyleSheet(GRAY)
        v.addWidget(note)

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
        self.batch_progress.setRange(0, 1)
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
        self.formant_ceiling_spin = QSpinBox()
        self.formant_ceiling_spin.setRange(3000, 20000)
        self.formant_ceiling_spin.setSingleStep(500)
        self.formant_ceiling_spin.setValue(7500)
        self.formant_ceiling_spin.setSuffix(" Hz")
        self.formant_ceiling_spin.setKeyboardTracking(False)
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
        self.freq_min_spin.setRange(0, 4900)
        self.freq_min_spin.setValue(200)
        self.freq_min_spin.setSuffix(" Hz")
        form.addRow("Freq min", self.freq_min_spin)

        self.freq_max_spin = QSpinBox()
        self.freq_max_spin.setRange(300, 40000)
        self.freq_max_spin.setValue(5000)
        self.freq_max_spin.setSuffix(" Hz")
        form.addRow("Freq max", self.freq_max_spin)
        # keep min < max at all times instead of failing at render time
        self.freq_min_spin.valueChanged.connect(lambda v: self.freq_max_spin.setMinimum(v + 100))
        self.freq_max_spin.valueChanged.connect(lambda v: self.freq_min_spin.setMaximum(v - 100))

        self.extend_freq_check = QCheckBox("Raise Freq max to fit boxes")
        self.extend_freq_check.setChecked(True)
        self.extend_freq_check.setToolTip("Off: the plot always stops at Freq max (boxes above it are cut off).")
        form.addRow(self.extend_freq_check)

        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["512", "1024", "2048", "4096"])
        self.fft_combo.setCurrentText("2048")
        form.addRow("FFT size", self.fft_combo)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["grayscale", "classic", "inferno"])
        form.addRow("Color scheme", self.color_combo)
        for spin in (self.padding_spin, self.freq_min_spin, self.freq_max_spin):
            spin.setKeyboardTracking(False)  # apply typed values on Enter / focus-out, not per keystroke
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
            lform.addRow(QLabel(elem), btn)
        return box

    def _build_image_box(self):
        box = QGroupBox("Image size (saved files)")
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

        # One persistent figure/canvas, reused (cleared + redrawn) on every Generate call, so the
        # on-screen size always matches the widget instead of jumping to a fixed size each time.
        self.fig = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, container)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.status_label = QLabel("Choose the recordings and the results folder to start.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        return container

    def _connect_live_options(self):
        """Redraw the current bout whenever an option that changes the picture or the measurements
        changes. (Image size only affects saved files, so it doesn't redraw.)"""
        for combo in (self.fft_combo, self.color_combo):
            combo.currentIndexChanged.connect(self._schedule_rerender)
        for spin in (self.formant_ceiling_spin, self.padding_spin, self.freq_min_spin, self.freq_max_spin):
            spin.valueChanged.connect(self._schedule_rerender)
        for check in (self.extend_freq_check, self.show_f0_check, self.show_f1_check, self.show_f2_check,
                      self.show_boxes_check):
            check.toggled.connect(self._schedule_rerender)
        for btn in [self.f0_color_btn, self.f1_color_btn, self.f2_color_btn, *self.element_color_btns.values()]:
            btn.changed.connect(self._schedule_rerender)
        self.bout_combo.currentIndexChanged.connect(self._schedule_rerender)

    # ---------- state helpers ----------

    def _set_status(self, text):
        self.status_label.setText(text)

    def _batch_running(self):
        return self._batch_thread is not None and self._batch_thread.isRunning()

    def _schedule_rerender(self, *_):
        if self._ready and self.last_render is not None and not self._batch_running():
            self._rerender_timer.start()

    def _flush_rerender(self):
        """Apply a pending redraw now, so Save writes exactly what the options say."""
        if self._rerender_timer.isActive():
            self._rerender_timer.stop()
            self._on_generate(auto=True)

    def _clear_figure(self):
        self._rerender_timer.stop()
        self.last_render = None
        self.fig.clf()
        self.canvas.draw_idle()
        self.save_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)

    # ---------- data source ----------

    def _browse_audio_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose the recordings folder", self.audio_row.text())
        if path:
            self.audio_row.edit.setText(path)
            self._on_paths_edited()

    def _browse_labels_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose the results folder", self.labels_row.text())
        if path:
            self.labels_row.edit.setText(path)
            self._on_paths_edited()

    def _browse_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_row.text())
        if path:
            self.output_row.edit.setText(path)
            self._on_output_edited()

    def _on_output_edited(self):
        if self.library is not None:
            self.library.output_dir = self.output_row.text() or self.library.output_dir

    def _open_output_folder(self):
        path = self.output_row.text()
        if path:
            os.makedirs(path, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _take_from_detection(self):
        if self._get_detection_paths is None:
            return
        audio, results = self._get_detection_paths()
        if audio:
            self.audio_row.edit.setText(audio)
        if results:
            self.labels_row.edit.setText(results)
        self._load_library(interactive=True)

    def notify_detection_finished(self):
        """Called by the window when a detection run has finished."""
        if self.available:
            self.data_hint.setText("A detection run just finished -- click 'Take from Detection tab' "
                                   "to look at its results here.")

    def _on_paths_edited(self):
        key = (self.audio_row.text(), self.labels_row.text())
        if self._ready and key != self._loaded_key:
            self._load_library(interactive=True)

    def _derive_output(self, labels):
        """Default output folder for a results folder; replaces an output we derived ourselves earlier,
        never one the user typed."""
        current = self.output_row.text()
        if labels and (not current or current == self._auto_output):
            self._auto_output = os.path.join(labels, "spectrograms")
            self.output_row.edit.setText(self._auto_output)

    def _load_library(self, interactive=True):
        audio, labels = self.audio_row.text(), self.labels_row.text()
        self._loaded_key = (audio, labels)
        if not audio or not labels:
            self._set_status("Choose the recordings folder and the results folder.")
            return False
        for name, path in (("recordings", audio), ("results", labels)):
            if not os.path.isdir(path):
                self._set_status(f"The {name} folder does not exist: {path}")
                if interactive:
                    QMessageBox.warning(self, "Folder not found", f"The {name} folder does not exist:\n{path}")
                return False
        self._derive_output(labels)
        try:
            self._set_status("Loading library…")
            library = ResultsLibrary(audio, labels, output_dir=self.output_row.text() or None)
        except Exception as exc:
            self._set_status(f"Could not load the library: {exc}")
            if interactive:
                QMessageBox.critical(self, "Failed to load library", f"Could not load the folders:\n{exc}")
            return False

        self.library = library
        self._clear_figure()
        animals = library.list_animals()
        for combo, items in ((self.animal_combo, animals), (self.batch_animal_combo, [ALL_ANIMALS] + animals)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.blockSignals(False)
        self._on_animal_changed(self.animal_combo.currentText())
        self._update_batch_summary()
        self._report_library(interactive)
        return True

    def _report_library(self, interactive):
        lib = self.library
        notes = []
        if lib.without_audio:
            notes.append(f"{len(lib.without_audio)} label file(s) without a recording of the same name")
        if lib.without_bouts:
            notes.append(f"{len(lib.without_bouts)} without any bout")
        if lib.unreadable:
            notes.append(f"{len(lib.unreadable)} unreadable")
        extra = f"  ({'; '.join(notes)})" if notes else ""

        if not lib.list_animals():
            self._set_status("No bouts found." + extra)
            if interactive and lib._label_index and lib.without_audio and len(lib.without_audio) == len(lib._label_index):
                QMessageBox.warning(
                    self, "No matching recordings",
                    f"Found {len(lib._label_index)} label file(s), but none has a recording of the same name "
                    f"anywhere under:\n{lib.audio_dir}\n\nA label file '<name>_labels.txt' belongs to '<name>.wav'.")
            elif interactive and not lib._label_index:
                QMessageBox.warning(self, "No label files", f"No *_labels.txt files found under:\n{lib.labels_dir}")
            return
        only_unknown = lib.list_animals() == [UNKNOWN_ANIMAL]
        self._set_status(
            f"Loaded {lib.n_bouts()} bout(s) in {lib.n_recordings()} recording(s), {len(lib.list_animals())} animal(s)"
            + (" -- no classification in these results, so all bouts are listed under 'unknown'." if only_unknown else ".")
            + extra)

    def _on_animal_changed(self, animal):
        self.recording_combo.blockSignals(True)
        self.recording_combo.clear()
        if animal and self.library:
            self.recording_combo.addItems(self.library.list_recordings(animal))
        self.recording_combo.blockSignals(False)
        self._on_recording_changed(self.recording_combo.currentText())

    def _on_recording_changed(self, rec):
        self.bout_combo.blockSignals(True)
        self.bout_combo.clear()
        animal = self.animal_combo.currentText()
        try:
            if animal and rec and self.library:
                for b in self.library.list_bouts(animal, rec):
                    t0, t1 = self.library.bout_span(animal, rec, b)
                    self.bout_combo.addItem(f"{b}   ({t0:.1f}–{t1:.1f} s)", b)
        except Exception as exc:
            QMessageBox.warning(self, "Could not read recording", str(exc))
        finally:
            self.bout_combo.blockSignals(False)
        self._schedule_rerender()

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
            extend_freq_to_boxes=self.extend_freq_check.isChecked(),
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

    def _on_generate(self, auto=False):
        if self._batch_running() or self.library is None:
            return
        animal = self.animal_combo.currentText()
        rec = self.recording_combo.currentText()
        bout = self.bout_combo.currentData()
        if not (animal and rec and bout is not None):
            if not auto:
                QMessageBox.warning(self, "Nothing selected", "Choose an animal, recording and bout first.")
            return

        self.generate_btn.setEnabled(False)
        self._set_status("Rendering…")
        self.status_label.repaint()
        try:
            _, meta = self.library.render(animal, rec, int(bout), self._current_settings(), fig=self.fig)
            self.canvas.draw()
            self.last_render = (animal, rec, int(bout), meta["measurements"])
            self.save_btn.setEnabled(True)
            self.export_csv_btn.setEnabled(True)
            self._set_status(f"Bout {bout}: {len(meta['measurements'])} element(s), "
                             f"{meta['t_start']:.2f}s – {meta['t_end']:.2f}s.")
        except (BoutNotFoundError, ValueError) as exc:
            if auto:
                self._set_status(f"Could not draw: {exc}")
            else:
                QMessageBox.warning(self, "Could not draw", str(exc))
        except Exception:
            if auto:
                self._set_status("Render failed — click Generate for details.")
            else:
                QMessageBox.critical(self, "Render failed", traceback.format_exc())
                self._set_status("Render failed.")
        finally:
            self.generate_btn.setEnabled(True)

    def _on_save(self):
        self._flush_rerender()
        if self.last_render is None:
            return
        animal, rec, bout, _ = self.last_render
        path = self.library.bout_path(animal, rec, bout, ".png")
        settings = self._current_settings()
        try:
            self.library.save_figure(self.fig, path, settings.image_dpi,
                                     width_in=settings.figure_width, height_in=settings.figure_height)
            self._set_status(f"Saved image to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_export_csv(self):
        self._flush_rerender()
        if self.last_render is None:
            return
        animal, rec, bout, measurements = self.last_render
        path = self.library.bout_path(animal, rec, bout, "_hz.csv")
        try:
            self.library.export_measurements(measurements, path, animal, rec, bout)
            self._set_status(f"Saved {len(measurements)} element(s) to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ---------- batch ----------

    def _batch_recordings(self):
        if not self.library:
            return []
        choice = self.batch_animal_combo.currentText()
        animals = self.library.list_animals() if choice == ALL_ANIMALS else [choice]
        return [(a, r) for a in animals if a for r in self.library.list_recordings(a)]

    def _update_batch_summary(self, *_):
        if not self.library:
            self.batch_summary.setText("Choose the recordings and results folders first.")
            return
        recs = self._batch_recordings()
        n_bouts = sum(len(self.library.list_bouts(a, r)) for a, r in recs)
        self.batch_summary.setText(f"{len(recs)} recording(s), {n_bouts} bout(s) from the detection results.")

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

        self._rerender_timer.stop()
        # The batch gets its own copy of the library so it never shares caches with the window.
        library = self.library.clone()
        library.output_dir = self.output_row.text() or library.output_dir
        self.batch_log.clear()
        self.batch_log.appendPlainText(f"Processing {len(recordings)} recording(s) → {library.output_dir}")
        self.batch_progress.setRange(0, 0)  # busy until the bout count is known

        self._batch_thread = QThread(self)
        self._batch_worker = BatchWorker(library, recordings, self._current_settings(), outputs,
                                         self.batch_workers_spin.value() or None)
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._set_batch_running(True)
        self._batch_thread.start()

    def _on_cancel_batch(self):
        if self._batch_worker:
            self._batch_worker.stop()
            self.batch_cancel_btn.setEnabled(False)
            self.batch_log.appendPlainText("Cancelling — finishing the bouts already in progress…")

    def _on_batch_progress(self, done, total, msg):
        self.batch_log.appendPlainText(msg)
        if total == 0:
            self.batch_progress.setRange(0, 0)
            self._set_status(f"Batch: {msg}")
            return
        self.batch_progress.setRange(0, total)
        self.batch_progress.setValue(done)
        self._set_status(f"Batch: {done}/{total} bouts")

    def _on_batch_finished(self, result):
        self._set_batch_running(False)
        state = "Cancelled" if result.get("cancelled") or result.get("stopped") else "Done"
        if result["total"] == 0:
            self.batch_progress.setRange(0, 1)
        self.batch_log.appendPlainText(
            f"{state}: {result['bouts']}/{result['total']} bout(s), {len(result['files'])} file(s) written, "
            f"{len(result['errors'])} error(s).")
        for e in result["errors"]:
            self.batch_log.appendPlainText(f"  ! {e}")
        self._set_status(f"Batch {state.lower()}: {len(result['files'])} file(s) in {self.output_row.text()}")

    def _on_batch_failed(self, tb):
        self._set_batch_running(False)
        self.batch_progress.setRange(0, 1)
        self.batch_log.appendPlainText("Batch failed.")
        QMessageBox.critical(self, "Batch failed", tb)

    def _set_batch_running(self, running):
        # Lock everything a batch depends on, so no option can change while it runs
        # (the batch uses the options as they were when it started).
        for w in [self.data_box, self.output_box, self.single_tab, self.batch_run_btn, self.batch_animal_combo,
                  self.batch_workers_spin, self.batch_png_check, self.batch_rec_csv_check, self.batch_all_csv_check,
                  *self.option_boxes]:
            w.setEnabled(not running)
        self.batch_cancel_btn.setEnabled(running)
        if not running:
            self.save_btn.setEnabled(self.last_render is not None)
            self.export_csv_btn.setEnabled(self.last_render is not None)

    def shutdown(self):
        """Stop a running batch (called when the window closes)."""
        if self.available and self._batch_running():
            self._batch_worker.stop()
            self._batch_thread.quit()
            self._batch_thread.wait(30000)  # finishes the current bout (a few seconds)

    # ---------- settings (saved in the same .cfg file as the detection tab) ----------

    def get_settings(self):
        """Everything worth restoring, as strings for the .cfg file. (Which animal / recording / bout is
        selected is not saved.)"""
        if not self.available:
            return {}
        s = {
            "recordings_dir": self.audio_row.text(),
            "results_dir": self.labels_row.text(),
            "output_dir": self.output_row.text(),
            "formant_ceiling": str(self.formant_ceiling_spin.value()),
            "padding": str(self.padding_spin.value()),
            "freq_min": str(self.freq_min_spin.value()),
            "freq_max": str(self.freq_max_spin.value()),
            "extend_freq_to_boxes": str(self.extend_freq_check.isChecked()),
            "fft_size": self.fft_combo.currentText(),
            "color_scheme": self.color_combo.currentText(),
            "show_f0": str(self.show_f0_check.isChecked()),
            "show_f1": str(self.show_f1_check.isChecked()),
            "show_f2": str(self.show_f2_check.isChecked()),
            "show_boxes": str(self.show_boxes_check.isChecked()),
            "color_f0": self.f0_color_btn.hex(),
            "color_f1": self.f1_color_btn.hex(),
            "color_f2": self.f2_color_btn.hex(),
            "image_width": str(self.export_width_spin.value()),
            "image_height": str(self.export_height_spin.value()),
            "image_dpi": str(self.dpi_spin.value()),
            "batch_png": str(self.batch_png_check.isChecked()),
            "batch_csv_per_recording": str(self.batch_rec_csv_check.isChecked()),
            "batch_combined_csv": str(self.batch_all_csv_check.isChecked()),
            "batch_workers": str(self.batch_workers_spin.value()),
        }
        for elem, btn in self.element_color_btns.items():
            s[f"color_{elem.lower()}"] = btn.hex()
        return s

    def set_settings(self, s):
        """Apply a settings dict from get_settings() (missing / invalid entries keep their current value),
        then load the library if both folders are set."""
        if not self.available:
            return
        was_ready, self._ready = self._ready, False  # no redraw / reload while values are being set

        def number(key, widget, cast):
            if key in s:
                try:
                    widget.setValue(cast(float(s[key])))
                except (TypeError, ValueError):
                    pass

        def flag(key, widget):
            if key in s:
                widget.setChecked(_to_bool(s[key]))

        def choice(key, combo):
            if s.get(key) is not None and combo.findText(s[key]) >= 0:
                combo.setCurrentText(s[key])

        try:
            # frequency limits depend on each other: open them up first, then set both, then re-apply
            self.freq_min_spin.setMaximum(100000)
            self.freq_max_spin.setMinimum(0)
            number("freq_min", self.freq_min_spin, int)
            number("freq_max", self.freq_max_spin, int)
            self.freq_max_spin.setMinimum(self.freq_min_spin.value() + 100)
            self.freq_min_spin.setMaximum(self.freq_max_spin.value() - 100)
            number("formant_ceiling", self.formant_ceiling_spin, int)
            number("padding", self.padding_spin, float)
            flag("extend_freq_to_boxes", self.extend_freq_check)
            choice("fft_size", self.fft_combo)
            choice("color_scheme", self.color_combo)
            flag("show_f0", self.show_f0_check)
            flag("show_f1", self.show_f1_check)
            flag("show_f2", self.show_f2_check)
            flag("show_boxes", self.show_boxes_check)
            for key, btn in (("color_f0", self.f0_color_btn), ("color_f1", self.f1_color_btn),
                             ("color_f2", self.f2_color_btn)):
                if key in s:
                    btn.set_hex(s[key])
            for elem, btn in self.element_color_btns.items():
                if f"color_{elem.lower()}" in s:
                    btn.set_hex(s[f"color_{elem.lower()}"])
            number("image_width", self.export_width_spin, float)
            number("image_height", self.export_height_spin, float)
            number("image_dpi", self.dpi_spin, int)
            flag("batch_png", self.batch_png_check)
            flag("batch_csv_per_recording", self.batch_rec_csv_check)
            flag("batch_combined_csv", self.batch_all_csv_check)
            number("batch_workers", self.batch_workers_spin, int)
            for key, row in (("recordings_dir", self.audio_row), ("results_dir", self.labels_row),
                             ("output_dir", self.output_row)):
                if key in s:
                    row.edit.setText(s[key])
        finally:
            self._ready = was_ready
        if self.audio_row.text() and self.labels_row.text():
            self._load_library(interactive=False)
