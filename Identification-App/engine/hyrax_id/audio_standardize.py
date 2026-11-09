"""Standardize input recordings before the pipeline: mono + target sample rate, 16-bit PCM wav.

Biologists sometimes record at 44.1 kHz, or save a stereo file in which only one channel carries signal.
The original files are NEVER modified: standardized copies (same file name) are written to a work folder
and the pipeline runs on those.

  * mono: channels whose peak is below `silence_threshold` are ignored; one active channel is kept as it is,
    several are averaged; if all are silent channel 1 is kept (the file is still processed and reported).
  * resample: polyphase FIR (scipy.signal.resample_poly) -- the same resampler the YOLO app uses to load audio.
  * files are handled by a thread pool: libsndfile I/O and numpy release the GIL, and threads (unlike
    processes) cannot re-import the GUI / torch in every worker or trip over a frozen build.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

DEFAULT_SILENCE_THRESHOLD = 0.001


@dataclass(frozen=True)
class StandardizeResult:
    name: str
    status: str            # "SUCCESS" | "SKIPPED" | "ERROR"
    message: str
    out_path: str | None = None


@dataclass
class StandardizeSummary:
    outputs: dict          # stem -> path of the standardized copy (successful files only)
    n_ok: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    cancelled: bool = False


def resolve_workers(workers) -> int:
    """None or 0 -> automatic (CPU cores minus one, at least 1)."""
    if not workers:
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(workers))


def to_mono(audio: np.ndarray, silence_threshold: float = DEFAULT_SILENCE_THRESHOLD):
    """(samples,) or (samples, channels) float audio -> (mono float32 array, short explanation)."""
    if audio.ndim == 1:
        return audio, "Already mono."
    if audio.ndim != 2:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")
    n_channels = audio.shape[1]
    if n_channels == 1:
        return audio[:, 0], "Single-channel audio."
    active = np.where(np.max(np.abs(audio), axis=0) > silence_threshold)[0]
    if len(active) == 0:
        return audio[:, 0], "All channels silent; keeping channel 1."
    if len(active) == 1:
        return audio[:, active[0]], f"Keeping active channel {active[0] + 1} of {n_channels}."
    return np.mean(audio[:, active], axis=1).astype(np.float32), f"Averaged {len(active)} active channels."


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    g = gcd(sr, target_sr)
    return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)


def standardize_file(src, dst, target_sr: int, silence_threshold: float = DEFAULT_SILENCE_THRESHOLD) -> StandardizeResult:
    """Read `src`, write the mono / target-rate / PCM_16 copy to `dst`. Never raises."""
    name = Path(src).name
    try:
        audio, sr = sf.read(str(src), dtype="float32")
        if audio.shape[0] == 0:
            return StandardizeResult(name, "SKIPPED", "Empty audio file.")
        audio, msg = to_mono(audio, silence_threshold)
        if sr != target_sr:
            audio = resample(audio, sr, target_sr)
            msg += f" Resampled {sr} Hz -> {target_sr} Hz."
        else:
            msg += f" Already {target_sr} Hz."
        sf.write(str(dst), audio, target_sr, subtype="PCM_16")
        return StandardizeResult(name, "SUCCESS", msg, str(dst))
    except Exception as exc:
        return StandardizeResult(name, "ERROR", f"{type(exc).__name__}: {exc}")


def standardize_files(files: dict, out_dir, target_sr: int, workers=None,
                      silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
                      cancel_event: threading.Event | None = None,
                      log: Callable[[str], None] = print) -> StandardizeSummary:
    """files: {stem: path}. Writes <out_dir>/<original file name> for each. Never touches the originals."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_workers = min(resolve_workers(workers), max(1, len(files)))
    log(f"[INFO] Standardizing {len(files)} file(s): mono, {target_sr} Hz, 16-bit ({n_workers} worker(s)).")
    summary = StandardizeSummary(outputs={})

    def job(stem, src):
        if cancel_event is not None and cancel_event.is_set():
            return stem, None
        return stem, standardize_file(src, out_dir / Path(src).name, target_sr, silence_threshold)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(job, stem, src) for stem, src in sorted(files.items())]
        done = 0
        for fut in as_completed(futures):
            stem, res = fut.result()
            if res is None:
                summary.cancelled = True
                continue
            done += 1
            if res.status == "SUCCESS":
                summary.n_ok += 1
                summary.outputs[stem] = res.out_path
                log(f"  [{done}/{len(files)}] {res.name}: {res.message}")
            else:
                summary.n_failed += res.status == "ERROR"
                summary.n_skipped += res.status == "SKIPPED"
                log(f"  [{done}/{len(files)}] [WARN] {res.name} {res.status.lower()}: {res.message}")
            if cancel_event is not None and cancel_event.is_set():
                pool.shutdown(wait=False, cancel_futures=True)
                summary.cancelled = True
                break
    log(f"[INFO] Standardized {summary.n_ok} of {len(files)} file(s)"
        + (f", {summary.n_skipped} skipped, {summary.n_failed} failed" if summary.n_skipped or summary.n_failed else "")
        + ".")
    return summary
