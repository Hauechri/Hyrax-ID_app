"""
Core spectrogram data-loading and rendering logic.

Headless (no Qt): returns a matplotlib Figure instead of calling
plt.show(), so any UI (or a script) can embed it. All pitch / formant
analysis goes through praat_measures (praat-parselmouth), so no Praat
executable is needed.
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from scipy.io.wavfile import read as wav_read
from scipy.signal import spectrogram as scipy_spectrogram

from praat_measures import (
    measure_f0, measure_formants, f0_contour, formant_tracks, FORMANT_CEILING_HZ,
)

warnings.filterwarnings("ignore")

DEFAULT_ELEMENT_COLOR_MAP = {
    "Wail": "#FF4500",
    "Chuck": "#1E90FF",
    "Trill": "#FF00FF",
}
DEFAULT_ELEMENT_COLOR = "#AAAAAA"

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
    """Loads an annotation table + audio library and renders bouts on demand.

    `app_dir` is the folder the default data paths are resolved against.
    `annotation_file` / `audio_base_dir` / `output_base_dir` point at the user's
    data library and default to the dataset layout next to the app if not given.
    """

    def __init__(self, app_dir, annotation_file=None, audio_base_dir=None, output_base_dir=None):
        self.app_dir = app_dir
        self.annotation_file = annotation_file or os.path.join(app_dir, "Songs analysed by Héloïse.xlsx")
        self.audio_base_dir = audio_base_dir or os.path.join(app_dir, "INPUT")
        self.output_base_dir = output_base_dir or os.path.join(app_dir, "OUTPUT")
        self.ann = pd.read_excel(self.annotation_file)
        self._wav_index = _index_wav_files(self.audio_base_dir)

    def list_singers(self):
        singers = []
        for singer in sorted(self.ann["Singer"].dropna().unique(), key=str):
            ann_s = self.ann[self.ann["Singer"] == singer]
            if any(str(sid) in self._wav_index for sid in ann_s["Song ID"].dropna().unique()):
                singers.append(str(singer))
        return singers

    def list_recordings(self, singer):
        ann_s = self.ann[self.ann["Singer"] == singer]
        return sorted(
            (sid for sid in ann_s["Song ID"].dropna().unique() if str(sid) in self._wav_index),
            key=str,
        )

    def list_bouts(self, singer, song_id):
        ann_song = self.ann[(self.ann["Singer"] == singer) & (self.ann["Song ID"] == song_id)]
        return sorted(int(b) for b in ann_song["Bout number "].dropna().unique())

    def render(self, animal_name, song_id, bout_number, settings: RenderSettings, fig=None):
        ann_song = self.ann[(self.ann["Singer"] == animal_name) & (self.ann["Song ID"] == song_id)]
        bout_rows = ann_song[ann_song["Bout number "] == bout_number]

        if ann_song.empty:
            available = sorted(self.ann[self.ann["Singer"] == animal_name]["Song ID"].dropna().unique(), key=str)
            raise BoutNotFoundError(
                f"No data found for animal='{animal_name}', song_id='{song_id}'.\n"
                f"Available recordings for {animal_name}: {available}"
            )
        if bout_rows.empty:
            available_bouts = sorted(ann_song["Bout number "].dropna().unique())
            raise BoutNotFoundError(
                f"Bout {bout_number} not found in {song_id}.\n"
                f"Available bouts: {[int(b) for b in available_bouts]}"
            )

        wav_path = self._wav_index.get(str(song_id))
        if not wav_path or not os.path.isfile(wav_path):
            raise FileNotFoundError(
                f"Audio file for recording '{song_id}' not found anywhere under: {self.audio_base_dir}"
            )

        fs, sig = _load_wav(wav_path)
        duration = len(sig) / fs
        t_start = max(0.0, float(bout_rows["Time start"].min()) - settings.padding_seconds)
        t_end = min(float(bout_rows["Time end "].max()) + settings.padding_seconds, duration)
        sig_crop = sig[int(t_start * fs): int(t_end * fs)]

        freqs, times_spec, Sxx_db = _build_spectrogram(sig_crop, fs, settings.fft_size)
        times_spec = times_spec + t_start

        # Overlay tracks (F0, F1, F2) are drawn inside each annotated element only —
        # outside the calls Praat would just be tracking background noise.
        tracks = {"F0": [], "F1": [], "F2": []}
        measurements = []
        for _, row in bout_rows.iterrows():
            r0, r1 = float(row["Time start"]), float(row["Time end "])
            elem = str(row["Element"]).strip()
            tracks["F0"].append(f0_contour(sig, fs, r0, r1))
            ft, (F1, F2, _F3) = formant_tracks(sig, fs, r0, r1, ceiling_hz=settings.formant_ceiling_hz)
            tracks["F1"].append((ft, F1))
            tracks["F2"].append((ft, F2))
            f0 = measure_f0(sig, fs, r0, r1, element=elem)
            fm = measure_formants(sig, fs, r0, r1, ceiling_hz=settings.formant_ceiling_hz)
            measurements.append({
                "element": elem, "time_start": r0, "time_end": r1,
                "annotated_hzmax": row.get("Hzmax", np.nan), "annotated_hzmin": row.get("Hzmin", np.nan),
                "auto_hzmax": f0["hz_max"], "auto_hzmin": f0["hz_min"],
                "f0_mean_hz": f0["f0_mean"], "f0_median_hz": f0["f0_median"],
                "F1_hz": fm["F1"], "F2_hz": fm["F2"], "F3_hz": fm["F3"],
                "formant_ceiling_hz": settings.formant_ceiling_hz,
            })

        shown_max = [m["auto_hzmax"] if settings.hz_source == "auto" else m["annotated_hzmax"] for m in measurements]
        if settings.hz_source == "auto" and settings.show_annotation_overlay:
            shown_max += [m["annotated_hzmax"] for m in measurements]
        max_box_hz = np.nanmax(np.asarray(shown_max, dtype=float)) if shown_max else np.nan
        fmax_display = max(
            settings.freq_max_hz,
            max_box_hz * 1.1 if pd.notna(max_box_hz) else settings.freq_max_hz,
        )

        cmap = CMAP_LOOKUP.get(settings.color_scheme, "gray_r")

        if fig is None:
            fig = Figure(figsize=(settings.figure_width, settings.figure_height), constrained_layout=True)
        else:
            fig.clf()
            fig.set_constrained_layout(True)
        ax = fig.add_subplot(111)

        fmask = (freqs >= settings.freq_min_hz) & (freqs <= fmax_display)
        sub = Sxx_db[fmask]
        ax.pcolormesh(
            times_spec, freqs[fmask], sub, cmap=cmap, shading="gouraud",
            vmin=np.percentile(sub, 5), vmax=np.percentile(sub, 99),
        )

        seen_elements = set()
        drew_outline = False
        if settings.show_hz_boxes:
            auto = settings.hz_source == "auto"
            for m in measurements:
                elem, r0, r1 = m["element"], m["time_start"], m["time_end"]
                color = settings.element_colors.get(elem, settings.default_element_color)
                ann_max, ann_min = m["annotated_hzmax"], m["annotated_hzmin"]
                hzmax_val, hzmin_val = (m["auto_hzmax"], m["auto_hzmin"]) if auto else (ann_max, ann_min)
                if pd.notna(hzmax_val) and pd.notna(hzmin_val):
                    _draw_hz_box(ax, r0, r1, hzmin_val, hzmax_val, color)
                    seen_elements.add(elem)
                if auto and settings.show_annotation_overlay and pd.notna(ann_max) and pd.notna(ann_min):
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
        source_label = "detected" if settings.hz_source == "auto" else "annotated"
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

    def export_measurements(self, measurements, out_path, animal_name, song_id, bout_number):
        df = pd.DataFrame(measurements)
        df.insert(0, "bout", int(bout_number))
        df.insert(0, "song_id", song_id)
        df.insert(0, "singer", animal_name)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)

    def default_output_path(self, animal_name, song_id, bout_number):
        out_dir = os.path.join(self.output_base_dir, animal_name, str(song_id))
        return os.path.join(out_dir, f"{animal_name}_{song_id}_bout_{int(bout_number):02d}.png")

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
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            finally:
                fig.set_size_inches(original_size)
                if fig.canvas is not None:
                    fig.canvas.draw_idle()
        else:
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
