"""
Core spectrogram data-loading and rendering logic.

Headless (no Qt): returns a matplotlib Figure instead of calling
plt.show(), so any UI (or a script) can embed it. All pitch / formant
analysis goes through praat_measures (praat-parselmouth), so no Praat
executable is needed.
"""
import os
import pickle
import sys
import tempfile
import warnings
from concurrent.futures import FIRST_COMPLETED, CancelledError, ProcessPoolExecutor, wait
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from scipy.io.wavfile import read as wav_read
from scipy.signal import spectrogram as scipy_spectrogram

from call_detection import detect_calls, group_into_bouts
from praat_measures import analyse_element, FORMANT_CEILING_HZ

warnings.filterwarnings("ignore")

DEFAULT_ELEMENT_COLOR_MAP = {
    "Wail": "#FF4500",
    "Chuck": "#1E90FF",
    "Snort": "#FF00FF",
    "Call": "#FF8C00",
}
DEFAULT_ELEMENT_COLOR = "#AAAAAA"
# constrained_layout already fits the margins, so no bbox_inches="tight" (it draws the
# figure twice); light PNG compression is ~4x faster to write for a slightly bigger file.
_SAVE_KW = {"pil_kwargs": {"compress_level": 3}}

DETECTED_ELEMENT = "Call"
BATCH_CHUNK = 4             # bouts per batch task: small enough that Cancel reacts within seconds   # element label for automatically detected calls (type unknown)

MEASUREMENT_COLUMNS = [
    "element", "time_start", "time_end", "boundaries",
    "annotated_hzmax", "annotated_hzmin", "auto_hzmax", "auto_hzmin",
    "f0_mean_hz", "f0_median_hz", "F1_hz", "F2_hz", "F3_hz", "formant_ceiling_hz",
]

CMAP_LOOKUP = {
    "grayscale": "gray_r",
    "classic": "jet",
    "inferno": "inferno",
}


class RenderSettings:
    def __init__(
        self,
        padding_seconds=1.0,
        freq_min_hz=200,
        freq_max_hz=5000,
        show_F0=True,
        show_F1=True,
        show_F2=True,
        show_hz_boxes=True,
        extend_freq_to_boxes=True,
        hz_source="auto",
        show_annotation_overlay=True,
        formant_ceiling_hz=FORMANT_CEILING_HZ,
        color_scheme="grayscale",
        fft_size=2048,
        color_f0="#00FF00",
        color_f1="#00FFFF",
        color_f2="#FFD700",
        element_colors=None,
        default_element_color=DEFAULT_ELEMENT_COLOR,
        figure_width=14,
        figure_height=6,
        image_dpi=150,
    ):
        self.padding_seconds = padding_seconds
        self.freq_min_hz = freq_min_hz
        self.freq_max_hz = freq_max_hz
        self.show_F0 = show_F0
        self.show_F1 = show_F1
        self.show_F2 = show_F2
        self.show_hz_boxes = show_hz_boxes
        # Raise the top of the plot above freq_max_hz when a drawn box would be cut off.
        self.extend_freq_to_boxes = extend_freq_to_boxes
        # "auto": detect HzMin / HzMax from the audio; "annotation": use the Excel values.
        self.hz_source = hz_source
        # In "auto" mode, also outline the annotated (ground-truth) box for comparison.
        self.show_annotation_overlay = show_annotation_overlay
        # Praat Burg formant ceiling used for the per-element F1–F3 in the CSV export.
        self.formant_ceiling_hz = formant_ceiling_hz
        self.color_scheme = color_scheme
        self.fft_size = fft_size
        self.color_f0 = color_f0
        self.color_f1 = color_f1
        self.color_f2 = color_f2
        self.element_colors = dict(element_colors) if element_colors else dict(DEFAULT_ELEMENT_COLOR_MAP)
        self.default_element_color = default_element_color
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.image_dpi = image_dpi


class BoutNotFoundError(ValueError):
    pass


_ANNOTATION_CACHE = {}


def _read_annotation(path):
    """Read the annotation Excel file, cached in memory by (path, size, mtime) so switching
    the annotation off and on again doesn't re-read it. Uses the fast calamine reader when
    installed (~10x faster than openpyxl on this file, identical result)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime)
    if key not in _ANNOTATION_CACHE:
        try:
            df = pd.read_excel(path, engine="calamine")
        except ImportError:
            df = pd.read_excel(path)
        _ANNOTATION_CACHE.clear()
        _ANNOTATION_CACHE[key] = df
    return _ANNOTATION_CACHE[key]


def _index_wav_files(audio_base_dir):
    """Map recording-ID (filename stem) -> full path, searched recursively.

    Recording IDs are unique filenames in this dataset, so the audio folder
    doesn't have to be organised as <folder>/<Animal>/<id>.wav — pointing
    straight at a single animal's folder, or any other layout, works too.
    """
    index = {}
    if not os.path.isdir(audio_base_dir):
        return index
    for root, _dirs, files in os.walk(audio_base_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                stem = os.path.splitext(f)[0]
                index.setdefault(stem, os.path.join(root, f))
    return index


def _load_wav(path):
    fs, sig = wav_read(path)
    if sig.ndim > 1:
        sig = sig[:, 0]
    sig = sig.astype(np.float64)
    sig -= sig.mean()
    mx = np.abs(sig).max()
    if mx > 0:
        sig /= mx
    return fs, sig


def _build_spectrogram(sig, fs, nperseg):
    noverlap = int(nperseg * 0.90)
    freqs, times, Sxx = scipy_spectrogram(
        sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
        scaling="density", mode="magnitude",
    )
    return freqs, times, 10 * np.log10(Sxx + 1e-10)


def _draw_hz_box(ax, t0, t1, hzmin, hzmax, color):
    ax.add_patch(mpatches.Rectangle(
        (t0, hzmin), t1 - t0, hzmax - hzmin,
        linewidth=2, edgecolor=color, facecolor=color, alpha=0.35, zorder=3,
    ))
    outline = [pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()]
    ax.plot([t0, t1], [hzmax, hzmax], color=color, lw=2.5, zorder=4, path_effects=outline)
    ax.plot([t0, t1], [hzmin, hzmin], color=color, lw=2.5, zorder=4, path_effects=outline)
    label_fx = [pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()]
    mid_t = (t0 + t1) / 2
    ax.text(mid_t, hzmax, f"{int(hzmax)} Hz", color=color, fontsize=7.5,
            fontweight="bold", ha="center", va="bottom", zorder=5, path_effects=label_fx)
    ax.text(mid_t, hzmin, f"{int(hzmin)} Hz", color=color, fontsize=7.5,
            fontweight="bold", ha="center", va="top", zorder=5, path_effects=label_fx)


def _draw_annotation_outline(ax, t0, t1, hzmin, hzmax, color):
    ax.add_patch(mpatches.Rectangle(
        (t0, hzmin), t1 - t0, hzmax - hzmin,
        linewidth=1.5, linestyle="--", edgecolor=color, facecolor="none", zorder=4,
        path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
    ))


class SpectrogramProject:
    """An audio library (folder of .wav files), optionally with an annotation table.

    With an annotation file, bouts and element start / end times come from it
    (and the annotated HzMin / HzMax are kept for comparison). Without one, calls
    are detected automatically (call_detection) and grouped into bouts by pauses;
    the animal is then the name of the sub-folder the .wav file sits in.
    """

    def __init__(self, audio_base_dir, annotation_file=None, output_base_dir=None):
        self.audio_base_dir = audio_base_dir
        self.annotation_file = annotation_file
        self.output_base_dir = output_base_dir or os.path.join(os.path.dirname(audio_base_dir), "OUTPUT")
        self.ann = _read_annotation(annotation_file) if annotation_file else None
        self._wav_index = _index_wav_files(audio_base_dir)
        self._detected_bouts = {}
        self._audio_cache = (None, None)
        self._measure_cache = {}

    def clone(self):
        """Same library (annotation table + file index shared read-only), separate caches —
        so a background batch never touches the caches the window is using."""
        other = object.__new__(SpectrogramProject)
        other.__dict__.update(self.__dict__)
        other._detected_bouts = dict(self._detected_bouts)
        other._audio_cache = (None, None)
        other._measure_cache = {}
        return other

    @property
    def uses_annotation(self):
        return self.ann is not None

    # ---------- library ----------

    def _folder_animal(self, song_id):
        rel = os.path.relpath(os.path.dirname(self._wav_index[song_id]), self.audio_base_dir)
        return "unknown" if rel in (".", "") else rel.split(os.sep)[0]

    def list_singers(self):
        if not self.uses_annotation:
            return sorted({self._folder_animal(s) for s in self._wav_index}, key=str)
        singers = []
        for singer in sorted(self.ann["Singer"].dropna().unique(), key=str):
            ann_s = self.ann[self.ann["Singer"] == singer]
            if any(str(sid) in self._wav_index for sid in ann_s["Song ID"].dropna().unique()):
                singers.append(str(singer))
        return singers

    def list_recordings(self, singer):
        if not self.uses_annotation:
            return sorted((s for s in self._wav_index if self._folder_animal(s) == singer), key=str)
        ann_s = self.ann[self.ann["Singer"] == singer]
        return sorted(
            (str(sid) for sid in ann_s["Song ID"].dropna().unique() if str(sid) in self._wav_index),
            key=str,
        )

    def list_bouts(self, singer, song_id):
        if not self.uses_annotation:
            return sorted(self._detect(song_id))
        ann_song = self._ann_rows(singer, song_id)
        return sorted(int(b) for b in ann_song["Bout number "].dropna().unique())

    def _ann_rows(self, singer, song_id):
        return self.ann[(self.ann["Singer"] == singer) & (self.ann["Song ID"].astype(str) == str(song_id))]

    def _audio(self, song_id):
        cached_id, data = self._audio_cache
        if cached_id != song_id:
            wav_path = self._wav_index.get(str(song_id))
            if not wav_path or not os.path.isfile(wav_path):
                raise FileNotFoundError(
                    f"Audio file for recording '{song_id}' not found anywhere under: {self.audio_base_dir}")
            data = _load_wav(wav_path)
            self._audio_cache = (song_id, data)
        return data

    def _detect(self, song_id):
        if song_id not in self._detected_bouts:
            fs, sig = self._audio(song_id)
            self._detected_bouts[song_id] = group_into_bouts(detect_calls(sig, fs))
        return self._detected_bouts[song_id]

    def bout_elements(self, singer, song_id, bout_number):
        """[{element, time_start, time_end, annotated_hzmax, annotated_hzmin}] for one bout."""
        if not self.uses_annotation:
            calls = self._detect(song_id).get(int(bout_number))
            if not calls:
                raise BoutNotFoundError(f"Bout {bout_number} not found in {song_id}.")
            return [dict(element=DETECTED_ELEMENT, time_start=t0, time_end=t1,
                         annotated_hzmax=np.nan, annotated_hzmin=np.nan) for t0, t1 in calls]

        ann_song = self._ann_rows(singer, song_id)
        if ann_song.empty:
            raise BoutNotFoundError(f"No data found for animal='{singer}', song_id='{song_id}'.")
        rows = ann_song[ann_song["Bout number "] == bout_number]
        if rows.empty:
            raise BoutNotFoundError(
                f"Bout {bout_number} not found in {song_id}.\n"
                f"Available bouts: {sorted(int(b) for b in ann_song['Bout number '].dropna().unique())}")
        return [dict(element=str(r["Element"]).strip() if pd.notna(r["Element"]) else "Unknown",
                     time_start=float(r["Time start"]),
                     time_end=float(r["Time end "]),
                     annotated_hzmax=r.get("Hzmax", np.nan), annotated_hzmin=r.get("Hzmin", np.nan))
                for _, r in rows.iterrows()]

    # ---------- analysis ----------

    def bout_span(self, singer, song_id, bout_number):
        """(start, end) in seconds of a bout, for labelling."""
        els = self.bout_elements(singer, song_id, bout_number)
        return min(e["time_start"] for e in els), max(e["time_end"] for e in els)

    def measure_bout(self, singer, song_id, bout_number, settings):
        """Per-element measurements for one bout (no plotting). Cached per bout and formant
        ceiling, so redrawing with other display options doesn't re-run Praat."""
        key = (str(singer), str(song_id), int(bout_number), float(settings.formant_ceiling_hz))
        if key not in self._measure_cache:
            self._measure_cache[key] = self._measure_bout(singer, song_id, bout_number, settings)
        return self._measure_cache[key]

    def _measure_bout(self, singer, song_id, bout_number, settings):
        fs, sig = self._audio(song_id)
        source = "annotation" if self.uses_annotation else "detected"
        out = []
        for e in self.bout_elements(singer, song_id, bout_number):
            f0, fm, f0_track, fm_tracks = analyse_element(
                sig, fs, e["time_start"], e["time_end"], element=e["element"],
                ceiling_hz=settings.formant_ceiling_hz)
            out.append({
                **e, "boundaries": source,
                "auto_hzmax": f0["hz_max"], "auto_hzmin": f0["hz_min"],
                "f0_mean_hz": f0["f0_mean"], "f0_median_hz": f0["f0_median"],
                "F1_hz": fm["F1"], "F2_hz": fm["F2"], "F3_hz": fm["F3"],
                "formant_ceiling_hz": settings.formant_ceiling_hz,
                # for plotting only (not exported: CSV columns are MEASUREMENT_COLUMNS)
                "_f0_track": f0_track, "_formant_tracks": fm_tracks,
            })
        return out

    def render(self, animal_name, song_id, bout_number, settings: RenderSettings, fig=None):
        measurements = self.measure_bout(animal_name, song_id, bout_number, settings)
        fs, sig = self._audio(song_id)
        duration = len(sig) / fs
        t_start = max(0.0, min(m["time_start"] for m in measurements) - settings.padding_seconds)
        t_end = min(max(m["time_end"] for m in measurements) + settings.padding_seconds, duration)
        sig_crop = sig[int(t_start * fs): int(t_end * fs)]

        freqs, times_spec, Sxx_db = _build_spectrogram(sig_crop, fs, settings.fft_size)
        times_spec = times_spec + t_start

        # Overlay tracks (F0, F1, F2) are drawn inside each element only —
        # outside the calls Praat would just be tracking background noise.
        tracks = {"F0": [], "F1": [], "F2": []}
        for m in measurements:
            tracks["F0"].append(m["_f0_track"])
            ft, (F1, F2, _F3) = m["_formant_tracks"]
            tracks["F1"].append((ft, F1))
            tracks["F2"].append((ft, F2))

        if settings.freq_max_hz <= settings.freq_min_hz:
            raise ValueError(f"Freq max ({settings.freq_max_hz} Hz) must be above Freq min "
                             f"({settings.freq_min_hz} Hz).")
        use_ann = settings.hz_source == "annotation" and self.uses_annotation
        compare = not use_ann and settings.show_annotation_overlay and self.uses_annotation
        # Freq max is the user's setting, extended only when they asked to fit the boxes.
        shown_max = []
        if settings.show_hz_boxes:
            shown_max += [m["annotated_hzmax"] if use_ann else m["auto_hzmax"] for m in measurements]
        if compare:
            shown_max += [m["annotated_hzmax"] for m in measurements]
        shown_max = np.asarray(shown_max, dtype=float)
        fmax_display = settings.freq_max_hz
        if settings.extend_freq_to_boxes and np.isfinite(shown_max).any():
            fmax_display = max(fmax_display, float(np.nanmax(shown_max)) * 1.1)

        cmap = CMAP_LOOKUP.get(settings.color_scheme, "gray_r")

        if fig is None:
            # Fixed-size export (batch): fixed margins. constrained_layout would draw the
            # whole figure twice on save, doubling the time per image.
            fig = Figure(figsize=(settings.figure_width, settings.figure_height))
            fig.subplots_adjust(left=0.065, right=0.985, bottom=0.09, top=0.93)
        else:
            fig.clf()
            fig.set_layout_engine("constrained")
        ax = fig.add_subplot(111)

        fmask = (freqs >= settings.freq_min_hz) & (freqs <= fmax_display)
        sub = Sxx_db[fmask]
        # imshow + bilinear looks like the old gouraud pcolormesh but renders ~50x faster
        # (gouraud took over a minute per image on long bouts). Both axes are evenly spaced.
        fsel = freqs[fmask]
        ax.imshow(
            sub, origin="lower", aspect="auto", interpolation="bilinear", cmap=cmap,
            extent=(times_spec[0], times_spec[-1], fsel[0], fsel[-1]),
            vmin=np.percentile(sub, 5), vmax=np.percentile(sub, 99),
        )

        # Filled boxes and comparison outlines are independent options.
        seen_elements = set()
        drew_outline = False
        for m in measurements:
            elem, r0, r1 = m["element"], m["time_start"], m["time_end"]
            color = settings.element_colors.get(elem, settings.default_element_color)
            ann_max, ann_min = m["annotated_hzmax"], m["annotated_hzmin"]
            hzmax_val, hzmin_val = (ann_max, ann_min) if use_ann else (m["auto_hzmax"], m["auto_hzmin"])
            if settings.show_hz_boxes and pd.notna(hzmax_val) and pd.notna(hzmin_val):
                _draw_hz_box(ax, r0, r1, hzmin_val, hzmax_val, color)
                seen_elements.add(elem)
            if compare and pd.notna(ann_max) and pd.notna(ann_min):
                _draw_annotation_outline(ax, r0, r1, ann_min, ann_max, color)
                drew_outline = True

        line_outline = [pe.Stroke(linewidth=3.5, foreground="black"), pe.Normal()]
        overlays = []
        if settings.show_F0:
            overlays.append(("F0", settings.color_f0, "F0 — fundamental frequency (pitch)"))
        if settings.show_F1:
            overlays.append(("F1", settings.color_f1, "F1 — 1st formant"))
        if settings.show_F2:
            overlays.append(("F2", settings.color_f2, "F2 — 2nd formant"))
        for key, color, _label in overlays:
            for t, v in tracks[key]:
                if len(t):
                    ln, = ax.plot(t, v, color=color, lw=2.0, zorder=6)
                    ln.set_path_effects(line_outline)

        ax.set_xlim(t_start, t_end)
        ax.set_ylim(settings.freq_min_hz, fmax_display)
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        ax.set_title(
            f"Bout {int(bout_number)}  |  Animal: {animal_name}  |  Recording: {song_id}",
            fontsize=13, fontweight="bold",
        )

        legend_handles = [mpatches.Patch(color=c, label=l) for _, c, l in overlays]
        source_label = "annotated" if use_ann else "detected"
        for elem in sorted(seen_elements):
            color = settings.element_colors.get(elem, settings.default_element_color)
            legend_handles.append(
                mpatches.Patch(facecolor=color, edgecolor=color, linewidth=2, alpha=0.6,
                                label=f"{elem}: HzMin / HzMax ({source_label})")
            )
        if drew_outline:
            legend_handles.append(
                mpatches.Patch(facecolor="none", edgecolor="#444444", linewidth=1.5, linestyle="--",
                                label="Annotated HzMin / HzMax (ground truth)")
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.85)

        return fig, {"t_start": t_start, "t_end": t_end, "measurements": measurements}

    # ---------- output ----------

    def recording_dir(self, animal_name, song_id):
        return os.path.join(self.output_base_dir, str(animal_name), str(song_id))

    def bout_path(self, animal_name, song_id, bout_number, suffix):
        """e.g. suffix '.png' or '_hz.csv' -> <out>/<animal>/<rec>/<animal>_<rec>_bout_NN<suffix>"""
        name = f"{animal_name}_{song_id}_bout_{int(bout_number):02d}{suffix}"
        return os.path.join(self.recording_dir(animal_name, song_id), name)

    @staticmethod
    def measurements_frame(measurements, animal_name, song_id, bout_number):
        df = pd.DataFrame(measurements, columns=MEASUREMENT_COLUMNS)
        df.insert(0, "bout", int(bout_number))
        df.insert(0, "song_id", song_id)
        df.insert(0, "singer", animal_name)
        return df

    def export_measurements(self, measurements, out_path, animal_name, song_id, bout_number):
        df = self.measurements_frame(measurements, animal_name, song_id, bout_number)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)

    def save_figure(self, fig, out_path, dpi, width_in=None, height_in=None):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if width_in and height_in:
            # The on-screen figure is continuously resized to match its Qt widget,
            # so export at an explicit, reproducible size instead of whatever the
            # window happens to be at the moment Save is clicked.
            original_size = fig.get_size_inches().copy()
            fig.set_size_inches(width_in, height_in)
            try:
                if fig.canvas is not None:
                    fig.canvas.draw()
                fig.savefig(out_path, dpi=dpi, **_SAVE_KW)
            finally:
                fig.set_size_inches(original_size)
                if fig.canvas is not None:
                    fig.canvas.draw_idle()
        else:
            fig.savefig(out_path, dpi=dpi, **_SAVE_KW)

    def run_batch(self, recordings, settings, save_png=True, csv_per_recording=True, combined_csv=True,
                  progress=None, should_stop=None, workers=None):
        """Process every bout of every (animal, song_id) in `recordings`.

        Bouts are processed in chunks of BATCH_CHUNK on `workers` processes (default: all
        cores but two, at most 6; 1 = in this process). progress(done, total, message) is
        called after each chunk (total 0 while still collecting bouts); should_stop() is
        polled between chunks — chunks already running finish, the rest is skipped.
        Returns {"bouts": n, "total": n, "files": [paths written], "errors": [...], "stopped": bool}.
        """
        errors, tasks = [], []
        for animal, song_id in recordings:
            if should_stop and should_stop():
                break
            if progress and not self.uses_annotation:
                progress(0, 0, f"Detecting calls in {song_id}…")
            try:
                bouts = self.list_bouts(animal, song_id)
            except Exception as exc:
                errors.append(f"{song_id}: {exc}")
                continue
            for i in range(0, len(bouts), BATCH_CHUNK):
                tasks.append((animal, song_id, bouts[i:i + BATCH_CHUNK]))
        total = sum(len(b) for _, _, b in tasks)
        if workers is None:
            workers = max(1, min(6, (os.cpu_count() or 2) - 2))
        workers = min(workers, len(tasks)) if tasks else 1

        results, done = [], 0
        stopped = bool(should_stop and should_stop())

        def collect(res):
            nonlocal done
            animal, song_id, rows = res
            results.append(res)
            done += len(rows)
            n_el = sum(len(df) for _, df, _, _ in rows if df is not None)
            bad = [err for *_, err in rows if err]
            errors.extend(bad)
            first, last = rows[0][0], rows[-1][0]
            span = f"bout {first}" if first == last else f"bouts {first}–{last}"
            msg = f"{animal} {song_id} {span}: {n_el} element(s)" + (f", {len(bad)} FAILED" if bad else "")
            if progress:
                progress(done, total, msg)

        if workers > 1 and not _workers_can_start():
            workers = 1
        if not stopped and tasks:
            shared = self.clone()   # separate caches from the window's project
            remaining = list(range(len(tasks)))
            if workers > 1:
                spec = _worker_spec(shared)
                try:
                    stopped, remaining, problem = _run_parallel(
                        tasks, settings, save_png, workers, spec, collect, should_stop)
                finally:
                    if spec["detected_path"]:
                        os.remove(spec["detected_path"])
                if problem and remaining and not stopped:
                    msg = f"Parallel processing failed ({problem}); continuing in one process."
                    errors.append(msg)
                    if progress:
                        progress(done, total, msg)
            if remaining and not stopped:
                _set_batch_project(shared)
                for i in remaining:
                    if should_stop and should_stop():
                        stopped = True
                        break
                    collect(_batch_task(*tasks[i], settings, save_png))

        # write CSVs in a fixed order (recording order as given, then bout), whatever the
        # order the workers finished in
        order = {rec: i for i, rec in enumerate(recordings)}
        rows = [(order.get((a, s_id), 0), a, s_id, bout, df, fl)
                for a, s_id, rr in results for bout, df, fl, _ in rr]
        rows.sort(key=lambda r: (r[0], r[3]))
        files = [f for *_, fl in rows for f in fl]
        frames = [r[4] for r in rows if r[4] is not None]
        if csv_per_recording:
            per_rec = {}
            for _, a, s_id, _, df, _ in rows:
                if df is not None:
                    per_rec.setdefault((a, s_id), []).append(df)
            for (animal, song_id), dfs in per_rec.items():
                path = os.path.join(self.recording_dir(animal, song_id), f"{animal}_{song_id}_hz.csv")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                pd.concat(dfs, ignore_index=True).to_csv(path, index=False)
                files.append(path)
        if combined_csv and frames:
            path = os.path.join(self.output_base_dir, f"batch_{datetime.now():%Y%m%d_%H%M%S}_hz.csv")
            os.makedirs(self.output_base_dir, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
            files.append(path)
        return {"bouts": len(frames), "total": total, "files": files, "errors": errors,
                "stopped": stopped}


# ---------- batch workers (module level so they can run in other processes) ----------

_BATCH_PROJECT = None
WORKER_START_TIMEOUT_S = 120   # workers import numpy / matplotlib / parselmouth first
NO_PROGRESS_TIMEOUT_S = 300    # a chunk takes ~10 s; nothing finishing for this long = stuck


def _ping():
    return os.getpid()


def _workers_can_start():
    """Worker processes re-import the main script (Windows 'spawn'). That only works for a
    script file or a frozen / compiled app — not e.g. for a script piped into python, where
    handing work to a dying worker can block. Then the batch runs in one process."""
    main = sys.modules.get("__main__")
    if getattr(sys, "frozen", False) or "__compiled__" in getattr(main, "__dict__", {}):
        return True
    path = getattr(main, "__file__", None)
    return bool(path) and os.path.isfile(path)


def _worker_spec(project):
    """What a worker needs to rebuild the library itself: paths only (a few hundred bytes —
    big pickles sent at process start can block if a worker dies early). Detected calls go
    through a temporary file so the workers don't detect them again."""
    path = None
    if project._detected_bouts:
        fd, path = tempfile.mkstemp(prefix="spectrogram_batch_", suffix=".pkl")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(project._detected_bouts, f)
    return dict(audio_base_dir=project.audio_base_dir, annotation_file=project.annotation_file,
                output_base_dir=project.output_base_dir, detected_path=path)


def _run_parallel(tasks, settings, save_png, workers, spec, collect, should_stop):
    """Run tasks on a process pool. Returns (stopped, indices still to do, problem or None).
    Never hangs: if the workers don't start or stop making progress, the pool is killed and
    the caller finishes the remaining tasks in its own process."""
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_batch_init, initargs=(spec,))
    stopped, problem, finished = False, None, set()
    try:
        try:
            pool.submit(_ping).result(timeout=WORKER_START_TIMEOUT_S)
        except Exception as exc:
            # set before returning: the finally block must kill the pool, not wait for it
            problem = f"workers did not start ({type(exc).__name__})"
            return False, list(range(len(tasks))), problem
        index = {pool.submit(_batch_task, *t, settings, save_png): i for i, t in enumerate(tasks)}
        pending = set(index)
        while pending:
            done_now, pending = wait(pending, timeout=NO_PROGRESS_TIMEOUT_S, return_when=FIRST_COMPLETED)
            if not done_now:
                problem = f"no progress for {NO_PROGRESS_TIMEOUT_S} s"
                break
            for fut in done_now:
                try:
                    collect(fut.result())
                    finished.add(index[fut])
                except CancelledError:
                    finished.add(index[fut])      # skipped on purpose (cancel)
                except Exception as exc:          # a worker process died
                    problem = repr(exc)
            if problem:
                break
            if not stopped and should_stop and should_stop():
                # cancel what hasn't started; keep collecting what is already running
                stopped = True
                for fut in pending:
                    fut.cancel()
        return stopped, [i for i in range(len(tasks)) if i not in finished], problem
    finally:
        if problem is None:
            pool.shutdown(wait=True, cancel_futures=True)
        else:
            procs = list(getattr(pool, "_processes", {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()


def _set_batch_project(project):
    global _BATCH_PROJECT
    _BATCH_PROJECT = project


def _batch_init(spec):
    """Worker-process initializer: rebuild the library from paths (see _worker_spec)."""
    warnings.filterwarnings("ignore")
    project = SpectrogramProject(spec["audio_base_dir"], annotation_file=spec["annotation_file"],
                                 output_base_dir=spec["output_base_dir"])
    if spec["detected_path"]:
        with open(spec["detected_path"], "rb") as f:
            project._detected_bouts = pickle.load(f)
    _set_batch_project(project)


def _batch_task(animal, song_id, bouts, settings, save_png):
    """Measure (and draw) a few bouts of one recording. Returns (animal, song_id,
    [(bout, DataFrame or None, [files], error or None)])."""
    p = _BATCH_PROJECT
    rows = []
    for bout in bouts:
        try:
            files = []
            if save_png:
                fig, meta = p.render(animal, song_id, bout, settings)
                path = p.bout_path(animal, song_id, bout, ".png")
                p.save_figure(fig, path, settings.image_dpi)
                files.append(path)
                measurements = meta["measurements"]
            else:
                measurements = p.measure_bout(animal, song_id, bout, settings)
            rows.append((bout, p.measurements_frame(measurements, animal, song_id, bout), files, None))
        except Exception as exc:
            rows.append((bout, None, [], f"{song_id} bout {bout}: {exc}"))
    p._measure_cache.clear()   # keep worker memory flat over a long batch
    return animal, song_id, rows
