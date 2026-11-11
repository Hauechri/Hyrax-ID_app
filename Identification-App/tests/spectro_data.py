"""Synthetic recordings + Audacity label files for the spectrogram viewer tests.

Truth (so tests can check the measurements): every call is a 5-harmonic tone whose F0 ramps linearly from f0a
to f0b, so its mean F0 is (f0a + f0b) / 2.
"""
import os

import numpy as np
import soundfile as sf

SR = 48000
# type, start, end, f0 start, f0 end
ELEMENTS = [("chuck", 1.0, 1.25, 1500, 2200), ("wail", 1.5, 2.1, 1200, 2600), ("chuck", 2.4, 2.6, 1800, 1400),
            ("wail", 6.0, 6.7, 1000, 2400), ("snort", 7.2, 7.4, 2000, 2100)]


def call(f0a, f0b, dur, sr=SR, amp=0.4):
    t = np.arange(int(dur * sr)) / sr
    ph = 2 * np.pi * np.cumsum(np.linspace(f0a, f0b, len(t))) / sr
    x = sum((1 / k) * np.sin(k * ph) for k in range(1, 6))
    return amp * x * np.hanning(len(t)) / np.abs(x).max()


def song(sr=SR, seconds=12, seed=1):
    rng = np.random.default_rng(seed)
    y = 0.01 * rng.standard_normal(int(sr * seconds))
    for _, a, b, f0a, f0b in ELEMENTS:
        c = call(f0a, f0b, b - a, sr)
        i = int(a * sr)
        y[i:i + len(c)] += c
    return (y * 0.8).astype(np.float32)


def labels_text(first_animal="FB", second_animal="QY"):
    """Audacity labels like the detection tab writes them. animal None -> bare 'BOUT{n}' (no classification)."""
    def head(n, a, b, animal):
        return f"{a:.6f}\t{b:.6f}\tBOUT{n}" + (f"_{animal}_0.950" if animal else "") + "\n"
    el = lambda rows: "".join(f"{a:.6f}\t{b:.6f}\t{t}_0.900\n" for t, a, b, _, _ in rows)
    return (head(1, 1.0, 2.6, first_animal) + el(ELEMENTS[:3]) + head(2, 6.0, 7.4, second_animal) + el(ELEMENTS[3:]))


def make_dataset(root):
    """audio/A/rec1.wav (mono 48 kHz, 2 bouts: FB, QY)      res/rec1_labels.txt
       audio/rec2.wav  (stereo 44.1 kHz, ONLY the right channel carries the song)  res/rec2_labels.txt (bare BOUTs)
       audio/rec4.wav  (silence)                                                  res/rec4_labels.txt (no bouts)
       res/rec3_labels.txt has no recording; res/notes.txt is a stray text file."""
    audio, res = os.path.join(root, "audio"), os.path.join(root, "res")
    os.makedirs(os.path.join(audio, "A"), exist_ok=True)
    os.makedirs(res, exist_ok=True)
    sf.write(os.path.join(audio, "A", "rec1.wav"), song(), SR, subtype="PCM_16")
    s44 = song(44100)
    sf.write(os.path.join(audio, "rec2.wav"), np.stack([np.zeros_like(s44), s44], axis=1), 44100, subtype="PCM_16")
    sf.write(os.path.join(audio, "rec4.wav"), np.zeros(SR * 3, dtype=np.float32), SR, subtype="PCM_16")
    open(os.path.join(res, "rec1_labels.txt"), "w", encoding="utf-8").write(labels_text("FB", "QY"))
    open(os.path.join(res, "rec2_labels.txt"), "w", encoding="utf-8").write(labels_text(None, None))
    open(os.path.join(res, "rec3_labels.txt"), "w", encoding="utf-8").write(labels_text("FB", "QY"))
    open(os.path.join(res, "rec4_labels.txt"), "w", encoding="utf-8").write("")
    open(os.path.join(res, "notes.txt"), "w", encoding="utf-8").write("not a label file")
    return audio, res
