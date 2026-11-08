"""Audio loading helpers (extracted unchanged from HYRAX_ID_predict.py)."""
import os
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf


def collect_wav_files(folder_path):
    """Collect all .wav files from the specified folder, keyed by stem."""
    return {
        Path(f).stem: os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.wav')
    }


def read_wave_file(filepath, peak_normalize=True):
    """Read a wave file as mono float32 (used for detection input).
    peak_normalize=True scales the peak to 1. False keeps the file's own level (integer files are
    scaled to [-1, 1] by their bit depth, like the YOLO app's audio loader)."""
    rate, data = wav.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono
    if peak_normalize:
        data = data.astype(np.float32)
        peak = np.max(np.abs(data)) if data.size else 0.0
        if peak > 0:
            data /= peak
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128
    elif np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    return data, rate


def load_audio(wav_path):
    """Read a wave file as (channels, T) float32, used for classifier input."""
    audio, sr = sf.read(wav_path)
    if len(audio.shape) == 1:
        audio = audio[None, :]
    else:
        audio = audio.T
    return audio.astype(np.float32), sr


def load_audio_segment(waveform, sr, start, end):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return waveform[:, start_sample:end_sample]
