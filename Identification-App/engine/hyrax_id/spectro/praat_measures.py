"""
Praat-based acoustic measurements (via the `praat-parselmouth` package; no
Praat executable needed).

F0 max / min per element
------------------------
Replicates how Héloïse measured Hzmax / Hzmin by hand in Avisoft
(FFT 256, Hamming, 50 % overlap): Hzmax is the upper edge of the F0 trace
at its highest point, Hzmin is the lower edge of the trace at its lowest
point. So we

  1. track F0 with Praat's autocorrelation pitch (floor 300 Hz, ceiling
     9 kHz), dropping octave jumps and correcting sub-harmonic errors
     against the spectrogram (the lowest *visible* band wins),
  2. ignore frames whose F0 ridge is > ENERGY_GATE_DB below the element's
     strongest frame (fading tails / background noise),
  3. take the highest / lowest remaining F0 value, and
  4. move from the ridge to the band edge on an Avisoft-like spectrogram
     (up for Hzmax, down for Hzmin), at most `edge_max_shift_hz`.

Profiles were tuned per element type on the 889 annotated elements that
have audio (song-wise train/test split). Held-out median |error| is
~100 Hz (max) / ~135 Hz (min), i.e. one to 1.5 of Avisoft's 94 Hz FFT
bins. Snorts are noisy and remain the least reliable.

Formants
--------
Héloïse's Praat settings: Burg, 3 formants, 0.03 s window. Her screenshot
shows a 5000 Hz ceiling, but her table values (F2/F3 up to ~14 kHz) fit
~7500–10000 Hz depending on the recording, so the ceiling is a parameter.
"""
import numpy as np
import parselmouth as pm
from parselmouth.praat import call
from scipy.signal import spectrogram, medfilt

PITCH_FLOOR_HZ = 300.0
PITCH_CEILING_HZ = 9000.0
PITCH_TIME_STEP = 0.0027          # ~ Avisoft hop at FFT 256 / 50 % overlap / 48 kHz
MIN_VOICING_STRENGTH = 0.2
SUBHARMONIC_FIX_DB = 20.0
ENERGY_GATE_DB = 25.0
AVISOFT_WINDOW_S = 256 / 48000.0  # Héloïse's FFT 256 at 48 kHz (5.3 ms), kept in seconds for other rates
ZERO_PAD = 4                      # window zero-padded 4x -> ~47 Hz grid


def avisoft_window(fs):
    """(nperseg, noverlap, nfft) of the Avisoft-like spectrogram at sample rate fs."""
    n = max(16, int(round(AVISOFT_WINDOW_S * fs)))
    return n, n // 2, n * ZERO_PAD
CONTEXT_PAD_S = 0.03

# edge_db_*: how far below the ridge peak the band edge is; trim_q > 0 uses the
# q / 1-q quantiles of F0 instead of its min / max.
DEFAULT_PROFILE = dict(edge_db_max=15.0, edge_db_min=15.0, edge_max_shift_hz=300.0, trim_q=0.0)
ELEMENT_PROFILES = {
    "Snort": dict(edge_db_max=0.0, edge_db_min=6.0, edge_max_shift_hz=200.0, trim_q=0.1),
}

FORMANT_COUNT = 3
FORMANT_WINDOW_S = 0.03
FORMANT_CEILING_HZ = 7500.0
FORMANT_CONTEXT_S = 0.5           # analyse a little around the element, like Praat's editor does


def _drop_octave_jumps(f0):
    if len(f0) < 5:
        return f0
    k = min(7, (len(f0) // 2) * 2 + 1)
    med = medfilt(f0, k)
    return np.where(np.abs(f0 - med) / med < 0.25, f0, np.nan)


def _fix_subharmonics(freqs, L, cols, fv):
    """Praat sometimes reports F0/2 or F0/3 when the real F0 band has no visible energy
    below it (e.g. a 2.3 kHz chuck tracked as 790 Hz). Like reading the spectrogram by eye,
    take the lowest multiple k*f0 (k = 1..4) whose level is within SUBHARMONIC_FIX_DB of the
    strongest multiple."""
    out = fv.copy()
    for j, (c, f0) in enumerate(zip(cols, fv)):
        ks = [k for k in range(1, 5) if k * f0 <= PITCH_CEILING_HZ]
        lev = []
        for k in ks:
            b = int(np.argmin(np.abs(freqs - k * f0)))
            lev.append(L[max(0, b - 2): b + 3, c].max())
        best = max(lev)
        out[j] = f0 * next(k for k, e in zip(ks, lev) if e >= best - SUBHARMONIC_FIX_DB)
    return out


def _band_edge(freqs, column_db, f_center, drop_db, upward, max_shift_hz):
    """From the ridge near `f_center`, walk up/down until the level falls `drop_db` below it."""
    k = int(np.argmin(np.abs(freqs - f_center)))
    lo, hi = max(0, k - 2), min(len(freqs), k + 3)
    k = lo + int(np.argmax(column_db[lo:hi]))
    peak, step, j = column_db[k], (1 if upward else -1), k
    while (0 <= j + step < len(freqs) and column_db[j + step] > peak - drop_db
           and abs(freqs[j + step] - freqs[k]) <= max_shift_hz):
        j += step
    return float(freqs[j])


def _track(sig, fs, t0, t1):
    """Cleaned F0 track inside [t0, t1] plus the Avisoft-like spectrogram it was checked on.
    Returns (times, f0, freqs, L_db, spectrogram_column_per_frame) or None if unvoiced."""
    a = max(0.0, t0 - CONTEXT_PAD_S)
    seg = sig[int(a * fs): int((t1 + CONTEXT_PAD_S) * fs)]
    if len(seg) < int(0.01 * fs):
        return None
    snd = pm.Sound(seg, sampling_frequency=fs, start_time=a)
    pitch = call(snd, "To Pitch (ac)", PITCH_TIME_STEP, PITCH_FLOOR_HZ, 15, "no",
                 0.03, MIN_VOICING_STRENGTH, 0.01, 0.35, 0.14, PITCH_CEILING_HZ)
    tt = pitch.xs()
    arr = pitch.selected_array
    f0, strength = arr["frequency"].astype(float), arr["strength"].astype(float)
    m = (tt >= t0) & (tt <= t1) & (f0 > 0) & (strength >= MIN_VOICING_STRENGTH)
    if m.sum() < 3:
        return None
    tv, fv = tt[m], _drop_octave_jumps(f0[m])
    ok = ~np.isnan(fv)
    tv, fv = tv[ok], fv[ok]
    if len(fv) < 3:
        return None

    nper, nover, nfft = avisoft_window(fs)
    freqs, ts, S = spectrogram(seg, fs=fs, window="hamming", nperseg=nper, noverlap=nover, nfft=nfft,
                               mode="magnitude")
    ts = ts + a
    L = 20 * np.log10(S + 1e-12)
    cols = np.array([int(np.argmin(np.abs(ts - t))) for t in tv])
    if SUBHARMONIC_FIX_DB > 0:
        fv = _fix_subharmonics(freqs, L, cols, fv)
        # frames the fix could not lift (noisy low end) are now isolated jumps; drop them
        fv = _drop_octave_jumps(fv)
        ok = ~np.isnan(fv)
        tv, fv, cols = tv[ok], fv[ok], cols[ok]
        if len(fv) < 3:
            return None
    return tv, fv, freqs, L, cols


def f0_contour(sig, fs, t0, t1):
    """(times, f0) of the cleaned F0 track over [t0, t1], for plotting. Empty arrays if unvoiced."""
    tr = _track(sig, fs, t0, t1)
    if tr is None:
        return np.array([]), np.array([])
    return tr[0], tr[1]


def measure_f0(sig, fs, t0, t1, element=None):
    """F0 measurements for one element: hz_max / hz_min (band edges, see module doc) plus the
    mean / median of the tracked F0 contour. Values are nan when no voiced F0 is found."""
    return _f0_from_track(_track(sig, fs, t0, t1), element)


def _f0_from_track(tr, element):
    out = dict(hz_max=np.nan, hz_min=np.nan, f0_mean=np.nan, f0_median=np.nan)
    if tr is None:
        return out
    _, fv, freqs, L, cols = tr
    ridge_db = np.array([L[int(np.argmin(np.abs(freqs - fv[j]))), cols[j]] for j in range(len(fv))])
    keep = ridge_db >= ridge_db.max() - ENERGY_GATE_DB
    fv, cols = fv[keep], cols[keep]
    out["f0_mean"], out["f0_median"] = float(np.mean(fv)), float(np.median(fv))

    prof = ELEMENT_PROFILES.get(element, DEFAULT_PROFILE)
    q = prof["trim_q"]
    if q > 0:
        kmax = int(np.argmin(np.abs(fv - np.quantile(fv, 1 - q))))
        kmin = int(np.argmin(np.abs(fv - np.quantile(fv, q))))
    else:
        kmax, kmin = int(np.argmax(fv)), int(np.argmin(fv))
    shift = prof["edge_max_shift_hz"]
    out["hz_max"] = _band_edge(freqs, L[:, cols[kmax]], fv[kmax], prof["edge_db_max"], True, shift)
    out["hz_min"] = _band_edge(freqs, L[:, cols[kmin]], fv[kmin], prof["edge_db_min"], False, shift)
    return out


def _formant_object(sig, fs, t0, t1, ceiling_hz, context_s):
    a = max(0.0, t0 - context_s)
    seg = sig[int(a * fs): int((t1 + context_s) * fs)]
    if len(seg) < int(FORMANT_WINDOW_S * fs):
        return None
    snd = pm.Sound(seg, sampling_frequency=fs, start_time=a)
    return call(snd, "To Formant (burg)", 0, FORMANT_COUNT, ceiling_hz, FORMANT_WINDOW_S, 50)


def measure_formants(sig, fs, t0, t1, ceiling_hz=FORMANT_CEILING_HZ):
    """Mean F1, F2, F3 (Hz) over [t0, t1], as Praat's editor 'Get first/second/third formant'."""
    out = {f"F{k}": np.nan for k in range(1, FORMANT_COUNT + 1)}
    formant = _formant_object(sig, fs, t0, t1, ceiling_hz, FORMANT_CONTEXT_S)
    if formant is None:
        return out
    for k in range(1, FORMANT_COUNT + 1):
        v = call(formant, "Get mean", k, t0, t1, "hertz")
        out[f"F{k}"] = float(v) if v == v else np.nan
    return out


def formant_tracks(sig, fs, t0, t1, ceiling_hz=FORMANT_CEILING_HZ):
    """(times, [F1, F2, F3]) formant tracks over [t0, t1], for plotting (nan where undefined)."""
    return _tracks_from_formant(_formant_object(sig, fs, t0, t1, ceiling_hz, 0.0), t0, t1)


def _tracks_from_formant(formant, t0, t1):
    if formant is None:
        return np.array([]), [np.array([])] * FORMANT_COUNT
    times = np.asarray(formant.xs())
    times = times[(times >= t0) & (times <= t1)]
    tracks = [np.array([formant.get_value_at_time(k, t) for t in times], dtype=float)
              for k in range(1, FORMANT_COUNT + 1)]
    return times, tracks


def analyse_element(sig, fs, t0, t1, element=None, ceiling_hz=FORMANT_CEILING_HZ):
    """Everything for one element in one pass (one pitch analysis, one formant analysis):
    (f0 measurements, {F1, F2, F3} means, (times, f0) track, (times, [F1, F2, F3]) tracks)."""
    tr = _track(sig, fs, t0, t1)
    f0 = _f0_from_track(tr, element)
    f0_track = (tr[0], tr[1]) if tr is not None else (np.array([]), np.array([]))

    fm = {f"F{k}": np.nan for k in range(1, FORMANT_COUNT + 1)}
    formant = _formant_object(sig, fs, t0, t1, ceiling_hz, FORMANT_CONTEXT_S)
    if formant is not None:
        for k in range(1, FORMANT_COUNT + 1):
            v = call(formant, "Get mean", k, t0, t1, "hertz")
            fm[f"F{k}"] = float(v) if v == v else np.nan
    return f0, fm, f0_track, _tracks_from_formant(formant, t0, t1)
