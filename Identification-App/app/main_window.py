"""
Main window: a QSplitter with input/output/model/settings controls on the
left (mirroring Alireza_Spectrogram_Viewer's controls-panel layout) and,
instead of a spectrogram canvas on the right, a live log console -- the
pipeline runs for minutes per file and only communicates via stdout, so the
log is the primary "what's happening, and what did it find" surface.

All four models (including the denoiser) are selected as file paths by the
user -- there is no online download step. Advanced pipeline parameters
(window/hop size, thresholds, etc.) are exposed as editable fields, and the
whole left-hand configuration can be saved to / loaded from a .cfg file.
"""
import configparser
import os
import time

from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

import paths
from results_model import ResultsModel
from worker import InferenceWorker

# Advanced pipeline parameters exposed in the UI, matching
# pipeline_runner.DEFAULT_PARAMS. "kind" drives which widget is built and
# how the value round-trips to/from the .cfg file.
#   int    -> QSpinBox
#   float  -> QDoubleSpinBox
#   bool   -> QCheckBox
ADVANCED_PARAM_SPECS = [
    ("window_size", "int", 48000, dict(minimum=1, maximum=10_000_000)),
    ("hop_size", "int", 24000, dict(minimum=1, maximum=10_000_000)),
    ("image_size", "int", 800, dict(minimum=1, maximum=100_000)),
    ("isMulti", "bool", True, {}),
    ("interarrival_threshold", "float", 0.724, dict(minimum=0.0, maximum=1000.0, decimals=3, singleStep=0.01)),
    ("detector_threshold", "float", 0.330, dict(minimum=0.0, maximum=1.0, decimals=3, singleStep=0.01)),
    ("context_windowsize", "int", 5, dict(minimum=0, maximum=100_000)),
    ("GB_threshold", "float", 0.3, dict(minimum=0.0, maximum=1.0, decimals=3, singleStep=0.01)),
    ("GB_scoreweight", "float", 0.3, dict(minimum=0.0, maximum=1.0, decimals=3, singleStep=0.01)),
    ("Denoiser_sequence_length", "int", 1, dict(minimum=1, maximum=100_000)),
    ("Denoiser_num_worker", "int", 0, dict(minimum=0, maximum=64)),
]

# Model path fields: (attribute prefix, label, default from paths.py)
MODEL_PATH_FIELDS = [
    ("detector", "Detector model", paths.DETECTOR_MODEL_PATH),
    ("garbage_filter", "Garbage filter model", paths.GARBAGE_FILTER_MODEL_PATH),
    ("animal_classifier", "Animal classifier model", paths.ANIMAL_CLASSIFIER_MODEL_PATH),
    ("denoiser", "Denoiser model", paths.DENOISER_ACA_MODEL_PATH),
]

CFG_SECTION = "hyraxid"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyrax-ID")
        self.resize(1150, 660)

        self._thread = None
        self._worker = None

        self._run_start_time = None
        self._file_slice_start_time = None
        self._latest_slice_progress = None
        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(1000)
        self._eta_timer.timeout.connect(self._update_eta_display)

        self._advanced_widgets = {}  # name -> widget
        self._model_path_edits = {}  # prefix -> QLineEdit

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_log_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 770])

    # -- left: controls panel --------------------------------------------

    def _build_controls_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(480)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)

        layout.addLayout(self._build_config_row())
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_models_group())
        layout.addWidget(self._build_device_group())
        layout.addWidget(self._build_advanced_group())

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._on_start)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.slice_progress_bar = QProgressBar()
        self.slice_progress_bar.setFormat("Slice %v/%m")
        self.slice_progress_bar.setVisible(False)
        layout.addWidget(self.slice_progress_bar)

        self.timing_row = QWidget()
        timing_layout = QHBoxLayout(self.timing_row)
        timing_layout.setContentsMargins(0, 0, 0, 0)
        self.elapsed_label = QLabel()
        self.remaining_label = QLabel()
        for lbl in (self.elapsed_label, self.remaining_label):
            lbl.setStyleSheet("color: #9a9a9a; font-size: 9pt;")
        timing_layout.addWidget(self.elapsed_label)
        timing_layout.addStretch(1)
        timing_layout.addWidget(self.remaining_label)
        self.timing_row.setVisible(False)
        layout.addWidget(self.timing_row)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        return scroll

    # -- config save/load --------------------------------------------------

    def _build_config_row(self):
        row = QHBoxLayout()
        load_button = QPushButton("Load settings...")
        load_button.clicked.connect(self._load_config)
        save_button = QPushButton("Save settings...")
        save_button.clicked.connect(self._save_config)
        row.addWidget(load_button)
        row.addWidget(save_button)
        return row

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "", "Config files (*.cfg)")
        if not path:
            return
        if not path.lower().endswith(".cfg"):
            path += ".cfg"

        cfg = configparser.ConfigParser()
        cfg[CFG_SECTION] = {}
        section = cfg[CFG_SECTION]

        section["input_mode"] = "single_file" if self.single_file_radio.isChecked() else "folder"
        section["input_path"] = self.input_path_edit.text()
        section["output_path"] = self.output_path_edit.text()
        section["device"] = self.device_combo.currentData()

        for prefix, _label, _default in MODEL_PATH_FIELDS:
            section[f"{prefix}_model_path"] = self._model_path_edits[prefix].text()

        for name, kind, _default, _kwargs in ADVANCED_PARAM_SPECS:
            widget = self._advanced_widgets[name]
            if kind == "bool":
                section[name] = str(widget.isChecked())
            else:
                section[name] = str(widget.value())

        try:
            with open(path, "w", encoding="utf-8") as f:
                cfg.write(f)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save settings:\n{exc}")
            return
        self.status_label.setText(f"Settings saved to {path}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "Config files (*.cfg)")
        if not path:
            return

        cfg = configparser.ConfigParser()
        try:
            with open(path, encoding="utf-8") as f:
                cfg.read_file(f)
        except (OSError, configparser.Error) as exc:
            QMessageBox.critical(self, "Load failed", f"Could not load settings:\n{exc}")
            return

        if CFG_SECTION not in cfg:
            QMessageBox.critical(self, "Load failed", f"'{path}' is not a Hyrax-ID settings file.")
            return
        section = cfg[CFG_SECTION]

        if section.get("input_mode") == "folder":
            self.folder_radio.setChecked(True)
        elif section.get("input_mode") == "single_file":
            self.single_file_radio.setChecked(True)
        self.input_path_edit.setText(section.get("input_path", self.input_path_edit.text()))
        self.output_path_edit.setText(section.get("output_path", self.output_path_edit.text()))

        device = section.get("device")
        if device is not None:
            idx = self.device_combo.findData(device)
            if idx != -1:
                self.device_combo.setCurrentIndex(idx)

        for prefix, _label, _default in MODEL_PATH_FIELDS:
            value = section.get(f"{prefix}_model_path")
            if value is not None:
                self._model_path_edits[prefix].setText(value)

        for name, kind, _default, _kwargs in ADVANCED_PARAM_SPECS:
            if name not in section:
                continue
            widget = self._advanced_widgets[name]
            try:
                if kind == "bool":
                    widget.setChecked(section.getboolean(name))
                elif kind == "int":
                    widget.setValue(section.getint(name))
                else:
                    widget.setValue(section.getfloat(name))
            except ValueError:
                continue  # leave that one field at its current value

        self.status_label.setText(f"Settings loaded from {path}")

    # -- input / output ------------------------------------------------

    def _build_input_group(self):
        group = QGroupBox("Input")
        outer = QVBoxLayout(group)

        mode_row = QHBoxLayout()
        self.single_file_radio = QRadioButton("Single file")
        self.folder_radio = QRadioButton("Folder")
        self.single_file_radio.setChecked(True)
        mode_row.addWidget(self.single_file_radio)
        mode_row.addWidget(self.folder_radio)
        mode_row.addStretch(1)
        outer.addLayout(mode_row)

        path_row = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_input)
        path_row.addWidget(self.input_path_edit)
        path_row.addWidget(browse_button)
        outer.addLayout(path_row)

        return group

    def _browse_input(self):
        if self.single_file_radio.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "Select a .wav file", "", "WAV files (*.wav)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select a folder of .wav files")
        if path:
            self.input_path_edit.setText(path)

    def _build_output_group(self):
        group = QGroupBox("Output folder")
        row = QHBoxLayout(group)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_output)
        row.addWidget(self.output_path_edit)
        row.addWidget(browse_button)
        return group

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select an output folder")
        if path:
            self.output_path_edit.setText(path)

    # -- models (all four are user-selected paths, incl. the denoiser) -----

    def _build_models_group(self):
        group = QGroupBox("Models")
        layout = QVBoxLayout(group)

        for prefix, label, default in MODEL_PATH_FIELDS:
            row_label = QLabel(label)
            layout.addWidget(row_label)

            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setText(default)
            browse_button = QPushButton("Browse...")
            browse_button.clicked.connect(lambda _checked=False, e=edit: self._browse_model_path(e))
            row.addWidget(edit)
            row.addWidget(browse_button)
            layout.addLayout(row)

            self._model_path_edits[prefix] = edit

        return group

    def _browse_model_path(self, line_edit):
        path, _ = QFileDialog.getOpenFileName(self, "Select model file")
        if path:
            line_edit.setText(path)

    def _build_device_group(self):
        group = QGroupBox("Device")
        layout = QVBoxLayout(group)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (recommended)", "auto")
        self.device_combo.addItem("CPU", "cpu")
        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except Exception:
            gpu_available = False
        gpu_label = "GPU (CUDA)" if gpu_available else "GPU (CUDA) -- not detected on this machine"
        self.device_combo.addItem(gpu_label, "cuda")
        layout.addWidget(self.device_combo)

        return group

    # -- advanced pipeline settings -----------------------------------------

    def _build_advanced_group(self):
        group = QGroupBox("Advanced settings")
        form = QFormLayout(group)

        for name, kind, default, kwargs in ADVANCED_PARAM_SPECS:
            if kind == "bool":
                widget = QCheckBox()
                widget.setChecked(default)
            elif kind == "int":
                widget = QSpinBox()
                widget.setMinimum(kwargs.get("minimum", 0))
                widget.setMaximum(kwargs.get("maximum", 1_000_000))
                widget.setValue(default)
            else:  # float
                widget = QDoubleSpinBox()
                widget.setDecimals(kwargs.get("decimals", 3))
                widget.setMinimum(kwargs.get("minimum", 0.0))
                widget.setMaximum(kwargs.get("maximum", 1.0))
                widget.setSingleStep(kwargs.get("singleStep", 0.01))
                widget.setValue(default)

            form.addRow(name, widget)
            self._advanced_widgets[name] = widget

        return group

    def _collect_param_overrides(self):
        overrides = {}
        for name, kind, _default, _kwargs in ADVANCED_PARAM_SPECS:
            widget = self._advanced_widgets[name]
            if kind == "bool":
                overrides[name] = widget.isChecked()
            elif kind == "int":
                overrides[name] = widget.value()
            else:
                overrides[name] = widget.value()
        return overrides

    # -- right: log panel --------------------------------------------------

    def _build_log_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        self.headline_label = QLabel("")
        self.headline_label.setWordWrap(True)
        self.headline_label.setStyleSheet("font-weight: bold; font-size: 13pt;")
        layout.addWidget(self.headline_label)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(right_splitter)

        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.addWidget(QLabel("Log"))
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(10000)
        self.log_console.setStyleSheet("font-family: Consolas, monospace;")
        log_layout.addWidget(self.log_console)
        right_splitter.addWidget(log_container)

        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_layout.addWidget(QLabel("Results"))
        self.results_model = ResultsModel(self)
        self.results_table = QTableView()
        self.results_table.setModel(self.results_model)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        results_layout.addWidget(self.results_table)
        right_splitter.addWidget(results_container)

        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 1)

        return container

    def _append_log(self, line):
        bar = self.log_console.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 2
        self.log_console.appendPlainText(line)
        if at_bottom:
            bar.setValue(bar.maximum())

    # -- run -----------------------------------------------------------

    def _on_start(self):
        input_path = self.input_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()

        if not input_path or not output_path:
            QMessageBox.warning(self, "Missing input", "Please select an input and an output folder.")
            return

        missing = [
            label
            for prefix, label, _default in MODEL_PATH_FIELDS
            if not self._model_path_edits[prefix].text().strip()
            or not os.path.isfile(self._model_path_edits[prefix].text().strip())
        ]
        if missing:
            QMessageBox.warning(
                self,
                "Missing model file(s)",
                "Please point every model field at a valid file:\n" + "\n".join(missing),
            )
            return

        self._start_inference()

    def _start_inference(self):
        input_path = self.input_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()
        os.makedirs(output_path, exist_ok=True)

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Processing...")
        self.log_console.clear()
        self.results_model.clear()
        self.headline_label.setText("")
        self._file_summaries = {}
        self.slice_progress_bar.setVisible(True)
        self.slice_progress_bar.setRange(0, 0)  # indeterminate until the first slice_progress arrives
        self.timing_row.setVisible(True)

        self._run_start_time = time.monotonic()
        self._file_slice_start_time = None
        self._latest_slice_progress = None
        self._update_eta_display()
        self._eta_timer.start()

        run_kwargs = dict(
            audio_path=input_path,
            output_path=output_path,
            detector_model_path=self._model_path_edits["detector"].text().strip(),
            garbage_filter_model_path=self._model_path_edits["garbage_filter"].text().strip(),
            animal_classifier_model_path=self._model_path_edits["animal_classifier"].text().strip(),
            denoiser_model_path=self._model_path_edits["denoiser"].text().strip(),
            device_override=self.device_combo.currentData(),
            **self._collect_param_overrides(),
        )

        self._thread = QThread(self)
        self._worker = InferenceWorker(run_kwargs)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_line.connect(self._append_log)
        self._worker.slice_progress.connect(self._on_slice_progress)
        self._worker.bout_result.connect(self.results_model.add_bout_result)
        self._worker.file_finished.connect(self._on_file_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)

        self._thread.start()

    def _on_slice_progress(self, current, total):
        self.slice_progress_bar.setRange(0, total)
        self.slice_progress_bar.setValue(current)
        if current == 1:
            # A new file's detector pass just started -- time it from here
            # rather than from the whole run's start, since denoising and
            # model loading time isn't representative of the per-slice rate.
            self._file_slice_start_time = time.monotonic()
        self._latest_slice_progress = (current, total)

    @staticmethod
    def _format_seconds(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _update_eta_display(self):
        if self._run_start_time is None:
            return
        elapsed = time.monotonic() - self._run_start_time
        self.elapsed_label.setText(f"Elapsed {self._format_seconds(elapsed)}")

        remaining_text = "Estimating remaining time..."
        if self._latest_slice_progress and self._file_slice_start_time is not None:
            current, total = self._latest_slice_progress
            if current > 0:
                file_elapsed = time.monotonic() - self._file_slice_start_time
                remaining = (file_elapsed / current) * (total - current)
                remaining_text = f"~{self._format_seconds(remaining)} remaining"
        self.remaining_label.setText(remaining_text)

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self._eta_timer.stop()
            self.status_label.setText("Cancelling... (will stop after the current file)")

    def _on_error(self, traceback_text):
        self._append_log(f"[ERROR] {traceback_text}")
        self._eta_timer.stop()
        self.status_label.setText("Failed.")
        QMessageBox.critical(self, "Pipeline error", traceback_text)

    def _on_file_finished(self, stem, summary):
        self._file_summaries[stem] = summary
        parts = []
        for name, s in self._file_summaries.items():
            if s["top_animal"] is None:
                parts.append(f"{name}: no animal identified")
            else:
                parts.append(f"{name}: {s['top_animal']}")
        self.headline_label.setText("  |  ".join(parts))

    def _on_finished(self):
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.slice_progress_bar.setVisible(False)
        self.timing_row.setVisible(False)
        self._eta_timer.stop()
        if self.status_label.text() not in ("Failed.", "Cancelling... (will stop after the current file)"):
            self.status_label.setText("Done.")
        elif self.status_label.text() == "Cancelling... (will stop after the current file)":
            self.status_label.setText("Cancelled.")
        # Deterministic shutdown: the worker's run() has already returned by
        # the time `finished` fires, so quit()+wait() here return almost
        # immediately -- doing it here (rather than via a separate signal
        # connection) avoids a race where the thread never gets told to stop.
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._worker = None
        self._thread = None