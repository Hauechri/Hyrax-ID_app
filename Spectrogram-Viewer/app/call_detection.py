"""
Automatic call (element) detection for recordings without an annotation file.

Energy-based: on a 500–9000 Hz spectrogram, every frequency is referenced to
its own background level over the whole file (so constant low-frequency noise
or hum doesn't count as a call). Per frame, "tonality" = strongest band minus
the median band; a call starts when tonality rises DETECT_ON_DB above its
rolling baseline, hysteresis (DETECT_OFF_DB) sets the boundaries, short gaps
are merged and very short blips dropped. Each call must also peak
DETECT_PEAK_DB above baseline. Calls separated by more than BOUT_GAP_S form
separate bouts — 0.8 s splits Héloïse's annotated bouts with ~1 % error
either way — and bouts longer than MAX_BOUT_S are split at their largest pause.

Against Héloïse's annotated elements: ~66–76 % of elements found, start / end
within ~10–20 ms (median). Only ~30 % of detections match an annotated element
(partly calls that were not annotated, partly noise). Element type
(Wail / Chuck / Snort) is not classified.
"""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import spectrogram

BAND_HZ = (500.0, 9000.0)
DETECT_ON_DB = 4.0
DETECT_OFF_DB = 2.0
DETECT_PEAK_DB = 8.0              # a detection must reach this somewhere to count as a call
MERGE_GAP_S = 0.02
MIN_CALL_S = 0.03
BOUT_GAP_S = 0.8
MAX_BOUT_S = 6.0                  # annotated bouts are < 3 s in 95 % of cases
NOISE_PERCENTILE = 20
BASELINE_S = 1.0                  # rolling-median window for the tonality baseline


def detect_calls(sig, fs):
    """Return a list of (t_start, t_end) in seconds for every detected call."""
    f, t, S = spectrogram(sig, fs=fs, window="hamming", nperseg=256, noverlap=128, mode="psd")
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    L = 10 * np.log10(S[band] + 1e-20)
    L = uniform_filter1d(L, 3, axis=0)                              # a little smoothing across frequency
    Lw = L - np.percentile(L, NOISE_PERCENTILE, axis=1)[:, None]    # dB above each band's background
    # Tonality: strongest band minus the median band. Broadband noise lifts both, a call
    # (a narrow F0 band + harmonics) only the first. The max alone is useless: with ~170
    # noisy bins one of them always spikes.
    dt = t[1] - t[0]
    tonal = median_filter(Lw.max(axis=0) - np.median(Lw, axis=0), size=5)
    baseline = median_filter(tonal, size=max(3, int(round(BASELINE_S / dt))) | 1, mode="nearest")
    level = tonal - baseline

    calls, start = [], None
    for i, v in enumerate(level):
        if start is None and v >= DETECT_ON_DB:
            start = i
            while start > 0 and level[start - 1] >= DETECT_OFF_DB:
                start -= 1
        elif start is not None and v < DETECT_OFF_DB:
            calls.append([t[start] - dt / 2, t[i - 1] + dt / 2])
            start = None
    if start is not None:
        calls.append([t[start] - dt / 2, t[-1] + dt / 2])

    merged = []
    for c in calls:
        if merged and c[0] - merged[-1][1] < MERGE_GAP_S:
            merged[-1][1] = c[1]
        else:
            merged.append(c)
    out = []
    for a, b in merged:
        if b - a < MIN_CALL_S:
            continue
        # a call must also peak clearly above the baseline somewhere (drops weak noise blips
        # that would otherwise chain whole minutes together into one "bout")
        if level[(t >= a) & (t <= b)].max(initial=-np.inf) < DETECT_PEAK_DB:
            continue
        out.append((max(0.0, float(a)), float(b)))
    return out


def _split_long(calls, max_s):
    """Split a run of calls at its largest pause until every part is at most `max_s` long."""
    if len(calls) < 2 or calls[-1][1] - calls[0][0] <= max_s:
        return [calls]
    gaps = [calls[i + 1][0] - calls[i][1] for i in range(len(calls) - 1)]
    k = int(np.argmax(gaps)) + 1
    return _split_long(calls[:k], max_s) + _split_long(calls[k:], max_s)


def group_into_bouts(calls, gap_s=BOUT_GAP_S, max_bout_s=MAX_BOUT_S):
    """Split calls into bouts wherever the pause exceeds `gap_s` (and split bouts longer than
    `max_bout_s` at their largest pause). Returns {bout_number: [calls]}."""
    runs = []
    for i, c in enumerate(calls):
        if i == 0 or c[0] - calls[i - 1][1] > gap_s:
            runs.append([])
        runs[-1].append(c)
    parts = [p for r in runs for p in _split_long(r, max_bout_s)]
    return {n: p for n, p in enumerate(parts, start=1)}
