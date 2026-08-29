"""
QThread worker that runs the pipeline off the GUI thread and streams its
stdout into the log console via Qt signals.
"""
import contextlib
import io
import os
import re
import threading
import traceback

from PySide6.QtCore import QObject, Signal

from pipeline_runner import run_batch, parse_bout_results, list_input_stems

# Matches HYRAX_ID_predict.py's per-window detector progress line:
# print(f"  - Slice {slice_index + 1}/{len(slices)}", end='\r')
SLICE_PROGRESS_RE = re.compile(r"^\s*-\s*Slice\s+(\d+)/(\d+)\s*$")


class QtLogStream(io.TextIOBase):
    """Buffers write() calls and emits one line per callback call, splitting
    on both \\n and \\r (the pipeline uses \\r for in-place slice progress)."""

    def __init__(self, emit_line):
        super().__init__()
        self._emit_line = emit_line
        self._buffer = ""

    def write(self, s):
        self._buffer += s
        while True:
            idx_candidates = [i for i in (self._buffer.find("\n"), self._buffer.find("\r")) if i != -1]
            if not idx_candidates:
                break
            idx = min(idx_candidates)
            line, self._buffer = self._buffer[:idx], self._buffer[idx + 1:]
            if line:
                self._emit_line(line)
        return len(s)

    def flush(self):
        pass

    def flush_remaining(self):
        if self._buffer:
            self._emit_line(self._buffer)
            self._buffer = ""


class InferenceWorker(QObject):
    log_line = Signal(str)
    slice_progress = Signal(int, int)
    bout_result = Signal(dict)
    file_finished = Signal(str, dict)
    finished = Signal()
    error = Signal(str)

    def __init__(self, run_kwargs):
        super().__init__()
        self._run_kwargs = run_kwargs
        self.cancel_event = threading.Event()

    def request_cancel(self):
        self.cancel_event.set()

    def _handle_line(self, line):
        # Detector slice progress is high-frequency and not meaningful log
        # history -- route it to a progress bar instead of flooding the log.
        m = SLICE_PROGRESS_RE.match(line)
        if m:
            self.slice_progress.emit(int(m.group(1)), int(m.group(2)))
            return
        self.log_line.emit(line)

    def run(self):
        stream = QtLogStream(self._handle_line)
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                run_batch(cancel_event=self.cancel_event, **self._run_kwargs)
            stream.flush_remaining()
            self._emit_results()
        except Exception:
            stream.flush_remaining()
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()

    def _emit_results(self):
        output_path = self._run_kwargs["output_path"]
        audio_path = self._run_kwargs["audio_path"]

        for stem in list_input_stems(audio_path):
            label_path = os.path.join(output_path, f"{stem}_labels.txt")
            # run_pipeline() only writes a labels file when it found at
            # least one bout; a missing file means "no bouts detected",
            # not "this file wasn't processed" (list_input_stems already
            # limits us to files this run actually submitted).
            bouts = parse_bout_results(label_path) if os.path.isfile(label_path) else []

            if not bouts:
                self.log_line.emit(f"RESULT: {stem}: no bouts detected.")
                self.file_finished.emit(
                    stem, {"top_animal": None, "top_confidence": 0.0, "n_bouts": 0, "label_path": label_path}
                )
                continue

            for b in bouts:
                self.bout_result.emit(
                    {
                        "file": stem,
                        "bout_id": b["bout_id"],
                        "animal": b["animal"],
                        "confidence": b["confidence"],
                        "start": b["start"],
                        "end": b["end"],
                        "n_elements": len(b["elements"]),
                    }
                )

            # Highest-confidence bout is used to pick "the" answer, but the
            # confidence number itself isn't shown -- it's not meaningful to
            # someone reading the result and was confusing rather than useful.
            top = max(bouts, key=lambda b: b["confidence"])
            self.log_line.emit(f"RESULT: {stem}: identified as {top['animal']}, {len(bouts)} bout(s) total")
            self.file_finished.emit(
                stem,
                {
                    "top_animal": top["animal"],
                    "top_confidence": top["confidence"],
                    "n_bouts": len(bouts),
                    "label_path": label_path,
                },
            )
