"""
Main window: a QSplitter with input/output/run controls on the left (mirroring
Alireza_Spectrogram_Viewer's controls-panel layout) and, instead of a
spectrogram canvas on the right, a live log console -- the pipeline runs for
minutes per file and only communicates via stdout, so the log is the primary
"what's happening, and what did it find" surface.
"""
import os

from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
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
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

import paths
from downloader import DownloadWorker, denoiser_cache_path, denoiser_is_cached
from results_model import ResultsModel
from worker import InferenceWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyrax-ID")
        self.resize(1100, 640)

        self._thread = None
        self._worker = None
        self._download_thread = None
        self._download_worker = None
        self._chain_to_inference_after_download = False

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_log_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 740])

    # -- left: controls panel --------------------------------------------

    def _build_controls_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(460)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_denoiser_group())
        layout.addWidget(self._build_device_group())

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

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        return scroll

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

    def _build_denoiser_group(self):
        group = QGroupBox("Denoiser model")
        layout = QVBoxLayout(group)

        self.denoiser_status_label = QLabel()
        self.denoiser_status_label.setWordWrap(True)
        layout.addWidget(self.denoiser_status_label)

        self.denoiser_progress = QProgressBar()
        self.denoiser_progress.setVisible(False)
        layout.addWidget(self.denoiser_progress)

        self.denoiser_download_button = QPushButton()
        self.denoiser_download_button.clicked.connect(lambda: self._start_download(chain_to_inference=False))
        layout.addWidget(self.denoiser_download_button)

        self._refresh_denoiser_status()
        return group

    def _refresh_denoiser_status(self):
        if denoiser_is_cached():
            self.denoiser_status_label.setText(f"Ready ({denoiser_cache_path()})")
            self.denoiser_download_button.setText("Re-download")
        else:
            self.denoiser_status_label.setText("Not downloaded yet -- fetched automatically the first time you press Start, or you can fetch it now.")
            self.denoiser_download_button.setText("Download now")

    def _start_download(self, chain_to_inference):
        if self._download_thread is not None:
            return  # already downloading

        self._chain_to_inference_after_download = chain_to_inference
        self.denoiser_download_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.denoiser_progress.setVisible(True)
        self.denoiser_progress.setRange(0, 0)  # indeterminate until we know the total
        self.denoiser_status_label.setText("Downloading denoiser model...")

        self._download_thread = QThread(self)
        self._download_worker = DownloadWorker()
        self._download_worker.moveToThread(self._download_thread)

        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.finished.connect(self._on_download_finished)

        self._download_thread.start()

    def _on_download_progress(self, read, total):
        if total > 0:
            self.denoiser_progress.setRange(0, total)
            self.denoiser_progress.setValue(read)
            self.denoiser_status_label.setText(f"Downloading denoiser model... {read / 1e6:.1f} / {total / 1e6:.1f} MB")
        else:
            self.denoiser_status_label.setText(f"Downloading denoiser model... {read / 1e6:.1f} MB")

    def _on_download_error(self, message):
        QMessageBox.critical(self, "Download failed", f"Could not download the denoiser model:\n{message}")
        self._chain_to_inference_after_download = False

    def _on_download_finished(self):
        if self._download_thread is not None:
            self._download_thread.quit()
            self._download_thread.wait()
        self._download_thread = None
        self._download_worker = None

        self.denoiser_progress.setVisible(False)
        self.denoiser_download_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self._refresh_denoiser_status()

        if self._chain_to_inference_after_download and denoiser_is_cached():
            self._chain_to_inference_after_download = False
            self._start_inference()
        else:
            self._chain_to_inference_after_download = False

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

        if not denoiser_is_cached():
            self._start_download(chain_to_inference=True)
            return

        self._start_inference()

    def _start_inference(self):
        input_path = self.input_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()
        os.makedirs(output_path, exist_ok=True)

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Processing... (this can take several minutes per file)")
        self.log_console.clear()
        self.results_model.clear()
        self.headline_label.setText("")
        self._file_summaries = {}
        self.slice_progress_bar.setVisible(True)
        self.slice_progress_bar.setRange(0, 0)  # indeterminate until the first slice_progress arrives

        run_kwargs = dict(
            audio_path=input_path,
            output_path=output_path,
            detector_model_path=paths.DETECTOR_MODEL_PATH,
            garbage_filter_model_path=paths.GARBAGE_FILTER_MODEL_PATH,
            animal_classifier_model_path=paths.ANIMAL_CLASSIFIER_MODEL_PATH,
            denoiser_model_path=denoiser_cache_path(),
            device_override=self.device_combo.currentData(),
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

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling... (will stop after the current file)")

    def _on_error(self, traceback_text):
        self._append_log(f"[ERROR] {traceback_text}")
        self.status_label.setText("Failed.")
        QMessageBox.critical(self, "Pipeline error", traceback_text)

    def _on_file_finished(self, stem, summary):
        self._file_summaries[stem] = summary
        parts = []
        for name, s in self._file_summaries.items():
            if s["top_animal"] is None:
                parts.append(f"{name}: no animal identified")
            else:
                parts.append(f"{name}: {s['top_animal']} ({s['top_confidence']:.0%})")
        self.headline_label.setText("  |  ".join(parts))

    def _on_finished(self):
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.slice_progress_bar.setVisible(False)
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
