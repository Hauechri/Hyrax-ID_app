"""
End-to-end inference pipeline:

    original audio  ----------------------------------------\
                                                               \
    denoised audio -> YOLO detector -> GarbageCollector (XGB) -> Bout Aggregator -> TDNN Animal Classifier -> Audacity labels

For every matched pair of files (same stem) in `audio_folder` (original) and
`audio_denoised_folder` (denoised):
  1. Detect chuck/snort/wail "elements" on the DENOISED audio with YOLO.
  2. Score each element with the GarbageCollector (XGBoost) to drop false positives.
  3. Group surviving elements into "bouts" based on interarrival time.
  4. For each bout, cut out its elements from the ORIGINAL audio, compute
     log-mel spectrograms, and run the TDNN classifier to get an animal ID.
  5. Write one Audacity label file per input file containing:
       - a label per bout:   start / end / "BOUT{n}_{animal}_{conf:.3f}"
       - a label per element inside that bout: start / end / "{type}_{conf:.3f}"

NOTE ON THE CLASSIFIER CALL:
  I don't have model_tdnn.py, so `run_classifier_on_bout()` below is my best
  guess at TDNNBoutModel's forward signature, based on how `process_file()`
  (kept below, unused, for reference) assembles training samples:
  {"elements": [variable-length mel tensors], "lengths": [...], "label": ...}.
  I pad each bout's mel-spectrograms to the bout's max length, build a
  lengths tensor, and add a batch dimension of 1. If TDNNBoutModel expects
  something else (e.g. it does its own padding/collation internally, or
  wants a list of un-padded tensors), share model_tdnn.py and I'll fix just
  that function.
"""

import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import joblib
import torch
import torchaudio
import soundfile as sf
import librosa
import scipy.io.wavfile as wav
from scipy.ndimage import zoom
from ultralytics import YOLO

import matplotlib
matplotlib.use('Agg')

from model_tdnn import TDNNBoutModel


# =========================================================
# GLOBAL CONSTANTS
# =========================================================

LABEL_MAP = {"chuck": 0, "snort": 1, "wail": 2}

SR = 48000

SPEAKER_LIST = [
    'J9', 'Kashtan', 'M0', 'M9', 'O1', 'O7', 'P0', 'P1', 'P8', 'Q7',
    'R3', 'T0', 'T1', 'T9', 'U7', 'U9', 'W4', 'X0'
]
# Fixed: the original script built a single dict named ID_TO_SPEAKER but
# populated it as speaker -> idx (i.e. it was actually SPEAKER_TO_ID), and
# `process_file` referenced a SPEAKER_TO_ID that never existed. Both real
# mappings are defined here, derived from one sorted list so they can never
# drift out of sync with each other.
SPEAKER_TO_ID = {speaker: idx for idx, speaker in enumerate(sorted(SPEAKER_LIST))}
ID_TO_SPEAKER = {idx: speaker for speaker, idx in SPEAKER_TO_ID.items()}


# =========================================================
# AUDIO I/O
# =========================================================

def collect_wav_files(folder_path):
    """Collect all .wav files from the specified folder, keyed by stem."""
    return {
        Path(f).stem: os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.wav')
    }


def match_audio_pairs(audio_folder, audio_denoised_folder):
    """
    Pair up files that exist in BOTH folders under the same stem
    (original for classification, denoised for detection).
    """
    originals = collect_wav_files(audio_folder)
    denoised = collect_wav_files(audio_denoised_folder)

    common = sorted(set(originals) & set(denoised))
    missing_denoised = sorted(set(originals) - set(denoised))
    missing_original = sorted(set(denoised) - set(originals))

    if missing_denoised:
        print(f"[WARN] {len(missing_denoised)} original file(s) have no denoised counterpart, skipping: {missing_denoised}")
    if missing_original:
        print(f"[WARN] {len(missing_original)} denoised file(s) have no original counterpart, skipping: {missing_original}")

    return [(stem, originals[stem], denoised[stem]) for stem in common]


def read_wave_file(filepath):
    """Read a wave file and normalize the signal to [-1, 1] (used for detection input)."""
    rate, data = wav.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
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


def slice_audio_signal(signal, window_size, hop_size):
    """Slice audio signal into overlapping windows."""
    total_samples = len(signal)
    slices = []
    for start in range(0, total_samples - window_size + 1, hop_size):
        end = start + window_size
        slices.append((signal[start:end], start, end))
    return slices


# =========================================================
# SPECTROGRAM GENERATION (DETECTOR INPUT)
# =========================================================

def resize(input, image_size=(800, 800)):
    """Resize spectrogram to match model input size."""
    target_h, target_w = image_size
    src_h, src_w = input.shape
    zoom_factors = (target_h / src_h, target_w / src_w)
    return zoom(input, zoom=zoom_factors, order=3)


def SpectroDynamics(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 60, width: int = 9):
    """Compute SpectroDynamics: base, delta, and delta-delta spectrograms."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)

    delta = librosa.feature.delta(S_db, width=width, order=1)
    delta2 = librosa.feature.delta(S_db, width=width, order=2)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    return norm(S_db), norm(delta), norm(delta2)


def generate_input_image(y, sr, image_size):
    """Convert audio slice into a 3-channel (spec, delta, delta-delta) image for YOLO."""
    spec0, spec1, spec2 = SpectroDynamics(y, sr)

    spec0 = resize(spec0, image_size)
    spec1 = resize(spec1, image_size)
    spec2 = resize(spec2, image_size)

    def to_uint8(x):
        x = np.clip(x, x.min(), x.max())
        return ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)

    return np.dstack([to_uint8(spec0), to_uint8(spec1), to_uint8(spec2)])


def waveform_to_logmel(
    waveform,
    sr=48000,
    n_mels=64,
    n_fft=1024,
    hop_length=144,  # 3 ms at 48kHz
    win_length=1024,
    f_min=50,
    f_max=20000,
    eps=1e-6
):
    """Convert waveform (1, T) -> log-mel spectrogram (T_frames, n_mels), used for the classifier."""
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0
    )
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    # Failsafe: torch.stft's default center=True reflect-pads by n_fft//2 on
    # each side, which requires the input to already be longer than that pad.
    # Very short bout elements (a few ms) can be shorter than n_fft and blow
    # this up. Zero-pad up to n_fft samples so STFT always has enough signal.
    min_len = n_fft
    if waveform.shape[-1] < min_len:
        pad_amount = min_len - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    mel_spec = mel_spec_transform(waveform)      # (1, n_mels, T)
    log_mel = torch.log(mel_spec + eps)
    log_mel = log_mel.squeeze(0).transpose(0, 1)  # (T, n_mels)
    return log_mel

# =========================================================
# YOLO DETECTION -> AUDACITY-STYLE LABELS
# =========================================================

def run_inference(model, numpy_array, threshold=0.05):
    results = model(numpy_array, conf=threshold)
    return results[0]


def convert_boxes_to_audacity_labels(prediction, image_duration_sec, start_sample, sample_rate, image_width,
                                      model=None, isMulti=False):
    """Convert YOLO bounding boxes to (start_sec, end_sec, class_name, conf) tuples."""
    labels = []
    class_map = {0: "chuck", 1: "snort", 2: "wail"} if isMulti else {0: "event"}

    for box in prediction.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box

        box_start_time = (x1 / image_width) * image_duration_sec
        box_end_time = (x2 / image_width) * image_duration_sec

        global_start_time = (start_sample / sample_rate) + box_start_time
        global_end_time = (start_sample / sample_rate) + box_end_time

        if model is not None and hasattr(model, "names") and int(cls) in model.names:
            cls_name = str(model.names[int(cls)]).lower()
        else:
            cls_name = class_map.get(int(cls), f"class_{int(cls)}")

        labels.append((global_start_time, global_end_time, cls_name, float(conf)))

    return labels


def merge_overlapping_labels(label_list, overlap_threshold=0.5, min_fraction_keep=0.5):
    """
    Merge labels of the same class if they overlap by at least a given fraction,
    then resolve overlaps between different classes based on confidence.
    """
    def overlap_fraction(a_start, a_end, b_start, b_end):
        overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
        len_a, len_b = a_end - a_start, b_end - b_start
        denom = min(len_a, len_b)
        return overlap / denom if denom > 0 else 0

    class_groups = defaultdict(list)
    for start, end, cls, conf in label_list:
        class_groups[cls].append((start, end, conf))

    merged_labels = []
    for cls, intervals in class_groups.items():
        intervals.sort()
        current_start, current_end, current_confs = intervals[0][0], intervals[0][1], [intervals[0][2]]

        for start, end, conf in intervals[1:]:
            frac = overlap_fraction(current_start, current_end, start, end)
            if frac >= overlap_threshold:
                current_end = max(current_end, end)
                current_confs.append(conf)
            else:
                mean_conf = sum(current_confs) / len(current_confs)
                merged_labels.append((current_start, current_end, cls, mean_conf))
                current_start, current_end, current_confs = start, end, [conf]

        mean_conf = sum(current_confs) / len(current_confs)
        merged_labels.append((current_start, current_end, cls, mean_conf))

    merged_labels.sort(key=lambda x: x[0])

    final_labels = []
    for label in merged_labels:
        start, end, cls, conf = label
        keep = True
        for i, (s2, e2, c2, conf2) in enumerate(final_labels):
            if e2 <= start or s2 >= end:
                continue

            overlap_start = max(start, s2)
            overlap_end = min(end, e2)
            overlap_len = overlap_end - overlap_start
            if overlap_len <= 0:
                continue

            if conf > conf2:
                orig_len = e2 - s2
                if overlap_start <= s2 and overlap_end >= e2:
                    final_labels[i] = None
                elif overlap_start <= s2:
                    new_start = overlap_end
                    new_len = e2 - new_start
                    final_labels[i] = (new_start, e2, c2, conf2) if new_len >= min_fraction_keep * orig_len else None
                elif overlap_end >= e2:
                    new_end = overlap_start
                    new_len = new_end - s2
                    final_labels[i] = (s2, new_end, c2, conf2) if new_len >= min_fraction_keep * orig_len else None
                else:
                    final_labels[i] = None
            else:
                orig_len = end - start
                if overlap_start <= start and overlap_end >= end:
                    keep = False
                    break
                elif overlap_start <= start:
                    start = overlap_end
                    new_len = end - start
                    if new_len < min_fraction_keep * orig_len:
                        keep = False
                        break
                elif overlap_end >= end:
                    end = overlap_start
                    new_len = end - start
                    if new_len < min_fraction_keep * orig_len:
                        keep = False
                        break
                else:
                    keep = False
                    break

        final_labels = [lbl for lbl in final_labels if lbl is not None]
        if keep:
            final_labels.append((start, end, cls, conf))

    final_labels.sort(key=lambda x: x[0])
    return final_labels


def group_bouts(merged_labels, interarrival_threshold=1.0):
    """Group merged labels into bouts based on interarrival time."""
    if not merged_labels:
        return []

    merged_labels = sorted(merged_labels, key=lambda x: x[0])

    bouts = []
    bout_start = merged_labels[0][0]
    bout_end = merged_labels[0][1]
    bout_number = 1

    for i in range(1, len(merged_labels)):
        start, end, cls, conf = merged_labels[i]
        gap = start - bout_end
        if gap <= interarrival_threshold:
            bout_end = max(bout_end, end)
        else:
            bouts.append((bout_start, bout_end, "BOUT", float(bout_number)))
            bout_number += 1
            bout_start = start
            bout_end = end

    bouts.append((bout_start, bout_end, "BOUT", float(bout_number)))
    return bouts


# =========================================================
# GARBAGE-COLLECTOR FEATURES
# =========================================================

def convert_to_detections(merged_labels):
    detections = []
    for i, (start, end, cls, conf) in enumerate(merged_labels):
        detections.append({"start": start, "end": end, "conf": conf, "class_id": cls, "idx": i})
    return detections


def nanmean(x):
    return np.nan if len(x) == 0 else np.nanmean(x)


def nanstd(x):
    return np.nan if len(x) == 0 else np.nanstd(x)


def compute_features(detections, window):
    N = len(detections)

    starts = np.array([d["start"] for d in detections], dtype=np.float32)
    ends = np.array([d["end"] for d in detections], dtype=np.float32)
    confs = np.array([d["conf"] for d in detections], dtype=np.float32)
    classes = np.array([LABEL_MAP[d["class_id"]] for d in detections], dtype=np.float32)
    durations = ends - starts

    features = []

    for i in range(N):
        feat = {}
        current_start = starts[i]

        feat["center_conf"] = confs[i]
        feat["center_duration"] = durations[i]
        feat["center_class"] = classes[i]

        neighbor_confs = []
        neighbor_durations = []
        neighbor_gaps = []
        local_classes = []

        for k in range(-window, window + 1):
            if k == 0:
                continue
            j = i + k
            prefix = f"n{k}"

            if 0 <= j < N:
                dt = starts[j] - current_start
                feat[f"{prefix}_exists"] = 1.0
                feat[f"{prefix}_dt"] = dt
                feat[f"{prefix}_conf"] = confs[j]
                feat[f"{prefix}_duration"] = durations[j]
                feat[f"{prefix}_class"] = classes[j]
                feat[f"{prefix}_delta_conf"] = confs[j] - confs[i]

                neighbor_confs.append(confs[j])
                neighbor_durations.append(durations[j])
                neighbor_gaps.append(abs(dt))
                local_classes.append(classes[j])
            else:
                feat[f"{prefix}_exists"] = 0.0
                feat[f"{prefix}_dt"] = np.nan
                feat[f"{prefix}_conf"] = np.nan
                feat[f"{prefix}_duration"] = np.nan
                feat[f"{prefix}_class"] = np.nan
                feat[f"{prefix}_delta_conf"] = np.nan

        feat["window_conf_mean"] = nanmean(neighbor_confs)
        feat["window_conf_std"] = nanstd(neighbor_confs)
        feat["window_duration_mean"] = nanmean(neighbor_durations)
        feat["window_duration_std"] = nanstd(neighbor_durations)
        feat["window_gap_mean"] = nanmean(neighbor_gaps)
        feat["window_gap_std"] = nanstd(neighbor_gaps)
        feat["window_density"] = len(neighbor_gaps)

        all_confs = np.array(neighbor_confs + [confs[i]])
        conf_rank = np.sum(all_confs < confs[i])
        feat["conf_rank"] = conf_rank
        feat["conf_percentile"] = conf_rank / len(all_confs)
        feat["is_local_max_conf"] = 1.0 if confs[i] >= np.nanmax(all_confs) else 0.0
        feat["is_local_min_conf"] = 1.0 if confs[i] <= np.nanmin(all_confs) else 0.0

        if len(local_classes) > 0:
            class_counts = Counter(local_classes)
            majority_class = class_counts.most_common(1)[0][0]
            feat["same_class_ratio"] = np.sum(np.array(local_classes) == classes[i]) / len(local_classes)
            feat["is_majority_class"] = 1.0 if classes[i] == majority_class else 0.0
        else:
            feat["same_class_ratio"] = np.nan
            feat["is_majority_class"] = np.nan

        features.append(feat)

    return features


def features_to_matrix(feature_list, feature_keys):
    X = []
    for feat in feature_list:
        X.append([feat.get(key, np.nan) for key in feature_keys])
    return np.array(X, dtype=np.float32)


# =========================================================
# STAGE 1+2+3: DETECT -> FILTER -> GROUP (per single file)
# =========================================================

def detect_elements_for_file(
    detector,
    garbagecollector,
    trained_feature_keys,
    denoised_wav_path,
    window_size,
    hop_size,
    image_size,
    threshold,
    isMulti,
    interarrival_threshold,
    context_windowsize,
    GB_threshold,
    GB_scoreweight,
):
    """
    Run YOLO detection + GarbageCollector filtering + bout grouping on a
    single (denoised) wav file.

    Returns:
        elements (list of (start, end, cls_name, conf)) : surviving elements
        bouts    (list of (start, end, "BOUT", bout_number))
    """
    signal, sr = read_wave_file(denoised_wav_path)
    slices = slice_audio_signal(signal, window_size, hop_size)

    raw_labels = []
    for slice_index, (audio_slice, start, _) in enumerate(slices):
        print(f"  - Slice {slice_index + 1}/{len(slices)}", end='\r')
        input_img = generate_input_image(audio_slice, sr, (image_size, image_size))
        prediction = run_inference(detector, input_img, threshold=threshold)
        labels = convert_boxes_to_audacity_labels(
            prediction,
            image_duration_sec=window_size / sr,
            start_sample=start,
            sample_rate=sr,
            image_width=image_size,
            model=detector,
            isMulti=isMulti,
        )
        raw_labels.extend(labels)
    print()

    if not raw_labels:
        return [], []

    merged_labels = merge_overlapping_labels(raw_labels)

    detections = convert_to_detections(merged_labels)
    features = compute_features(detections, context_windowsize)
    X = features_to_matrix(features, trained_feature_keys)

    probs = garbagecollector.predict_proba(X)[:, 1]
    confs = np.array([f["center_conf"] for f in features])
    combined_score = GB_scoreweight * confs + (1 - GB_scoreweight) * probs

    remain_labels = [label for label, p in zip(merged_labels, combined_score) if p > GB_threshold]

    bout_labels = group_bouts(remain_labels, interarrival_threshold=interarrival_threshold)

    return remain_labels, bout_labels


# =========================================================
# STAGE 4: ASSIGN ELEMENTS TO BOUTS (in-memory, no text round-trip)
# =========================================================

def build_bout_data(elements, bouts):
    """
    elements: list of (start, end, cls_name, conf)
    bouts:    list of (start, end, "BOUT", bout_number)

    Returns a list of dicts:
        {bout_id, start, end, elements: [{start, end, type, confidence}, ...]}
    """
    bout_data = []

    for b_start, b_end, _, bout_number in bouts:
        bout_elements = []
        for start, end, cls, conf in elements:
            # overlap condition
            if not (end < b_start or start > b_end):
                bout_elements.append({
                    "start": start,
                    "end": end,
                    "type": cls,
                    "confidence": conf,
                })

        bout_elements.sort(key=lambda x: x["start"])

        bout_data.append({
            "bout_id": int(bout_number),
            "start": b_start,
            "end": b_end,
            "elements": bout_elements,
        })

    return bout_data


# =========================================================
# STAGE 5: ANIMAL CLASSIFIER
# =========================================================

def run_classifier_on_bout(model, mel_list, device):
    """
    mel_list: list of (T_i, n_mels) tensors, one per element in the bout
    (variable length, unpadded — matches training in train_tdnn.py, where
    `elements` is a list-of-lists of un-padded per-element tensors and the
    model is always called with batch size 1).

    TDNNBoutModel.forward(self, batch_elements) expects batch_elements to be
    a list of bouts, where each bout is itself a list of (T, F) tensors.
    Since we only ever classify one bout at a time here, we wrap mel_list in
    an outer list of length 1.
    """
    elements = [[e.to(device) for e in mel_list]]  # batch of size 1

    with torch.no_grad():
        logits = model(elements)  # (1, num_classes)
        probs = torch.softmax(logits, dim=-1)
        pred = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, pred].item())

    return pred, confidence


def predict_animals_for_file(original_waveform, sr, bout_data, model, device):
    """
    For each bout, extract its elements from the ORIGINAL (non-denoised)
    audio, compute log-mel spectrograms, and classify the bout.

    Adds "animal" and "animal_confidence" keys to each bout dict in place,
    and returns the (mutated) bout_data list.
    """
    for bout in bout_data:
        if not bout["elements"]:
            bout["animal"] = "unknown"
            bout["animal_confidence"] = 0.0
            continue

        mel_list = []
        for e in bout["elements"]:
            segment = load_audio_segment(original_waveform, sr, e["start"], e["end"])
            mel = waveform_to_logmel(segment, sr=sr)
            mel_list.append(mel)

        pred, confidence = run_classifier_on_bout(model, mel_list, device)

        bout["animal"] = ID_TO_SPEAKER.get(pred, f"unknown_id_{pred}")
        bout["animal_confidence"] = confidence

    return bout_data


# =========================================================
# STAGE 6: WRITE FULL AUDACITY LABEL FILE
# =========================================================

def save_full_audacity_labels(bout_data, output_path):
    """
    Writes one Audacity label file containing, per bout:
      - a bout-level label:    start / end / "BOUT{n}_{animal}_{conf:.3f}"
      - element-level labels:  start / end / "{type}_{conf:.3f}"
    """
    lines = []
    for bout in bout_data:
        lines.append((
            bout["start"],
            bout["end"],
            f"BOUT{bout['bout_id']}_{bout['animal']}_{bout['animal_confidence']:.3f}",
        ))
        for e in bout["elements"]:
            lines.append((e["start"], e["end"], f"{e['type']}_{e['confidence']:.3f}"))

    lines.sort(key=lambda x: x[0])

    with open(output_path, "w") as f:
        for start, end, label in lines:
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")


# =========================================================
# (Reference only, not used by the inference pipeline below)
# Kept for training-data prep; requires ground-truth *_labels.txt files.
# =========================================================

def parse_annotation_file(path):
    elements = []
    bouts = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            start, end, label = line.split("\t")
            start = float(start)
            end = float(end)

            if label.startswith("BOUT"):
                bout_id = int(float(label.split("_")[1]))
                bouts.append({"start": start, "end": end, "bout_id": bout_id})
            else:
                parts = label.split("_")
                elements.append({
                    "start": start,
                    "end": end,
                    "type": parts[0],
                    "confidence": float(parts[1]),
                })

    return elements, bouts


def assign_elements_to_bouts(elements, bouts):
    bout_data = []
    for i, bout in enumerate(bouts):
        b_start, b_end = bout["start"], bout["end"]
        bout_elements = [e for e in elements if not (e["end"] < b_start or e["start"] > b_end)]
        bout_elements = sorted(bout_elements, key=lambda x: x["start"])
        bout_data.append({"bout_id": i, "start": b_start, "end": b_end, "elements": bout_elements})
    return bout_data


def process_file(wav_path, txt_path):
    """Training-data prep helper. Not used for inference."""
    waveform, sr = load_audio(wav_path)
    assert sr == SR, f"Sample rate mismatch: {sr}"

    elements, bouts = parse_annotation_file(txt_path)
    bout_data = assign_elements_to_bouts(elements, bouts)

    base_name = Path(wav_path).stem
    stem = base_name.replace("_labels.txt", "")
    parts = stem.split("_")
    speaker = parts[0]

    label = SPEAKER_TO_ID[speaker]

    for bout in bout_data:
        element_images = [
            waveform_to_logmel(load_audio_segment(waveform, sr, e["start"], e["end"]), sr)
            for e in bout["elements"]
        ]
        sample = {
            "elements": element_images,
            "lengths": [x.shape[0] for x in element_images],
            "label": label,
        }
        return sample


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline(
    Detector_model_path,
    GarbageFilter_model_path,
    Animal_Classifier_model_path,
    audio_folder,
    audio_denoised_folder,
    output_path,
    window_size,
    hop_size,
    image_size,
    detector_threshold,
    isMulti,
    interarrival_threshold,
    context_windowsize,
    GB_threshold,
    GB_scoreweight,
    num_classes,
    input_dim=64,
    emb_dim=128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    detector = YOLO(Detector_model_path)
    print(f"[INFO] Detector loaded: {Detector_model_path}")

    gc_data = joblib.load(GarbageFilter_model_path)
    garbagecollector = gc_data["model"]
    trained_feature_keys = list(gc_data["feature_keys"])
    print(f"[INFO] GarbageCollector loaded: {GarbageFilter_model_path}")

    classifier = TDNNBoutModel(num_classes=num_classes, input_dim=input_dim, emb_dim=emb_dim)
    classifier.load_state_dict(torch.load(Animal_Classifier_model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    print(f"[INFO] Animal classifier loaded: {Animal_Classifier_model_path}")

    os.makedirs(output_path, exist_ok=True)

    pairs = match_audio_pairs(audio_folder, audio_denoised_folder)
    print(f"[INFO] Found {len(pairs)} matched (original, denoised) file pairs.")

    for file_index, (stem, original_path, denoised_path) in enumerate(pairs):
        print(f"\n[FILE {file_index + 1}/{len(pairs)}] Processing: {stem}")

        elements, bouts = detect_elements_for_file(
            detector,
            garbagecollector,
            trained_feature_keys,
            denoised_path,
            window_size,
            hop_size,
            image_size,
            detector_threshold,
            isMulti,
            interarrival_threshold,
            context_windowsize,
            GB_threshold,
            GB_scoreweight,
        )

        if not bouts:
            print(f"  [INFO] No bouts detected for {stem}, skipping.")
            continue

        bout_data = build_bout_data(elements, bouts)

        original_waveform, sr = load_audio(original_path)
        if sr != SR:
            print(f"  [WARN] {stem}: sample rate {sr} != expected {SR}")

        bout_data = predict_animals_for_file(original_waveform, sr, bout_data, classifier, device)

        label_output_path = os.path.join(output_path, f"{stem}_labels.txt")
        save_full_audacity_labels(bout_data, label_output_path)
        print(f"  ✓ Labels saved to {label_output_path}")


if __name__ == "__main__":
    # === SETTINGS ===
    window_size = 48000
    hop_size = 24000
    image_size = 800
    isMulti = True
    interarrival_threshold = 0.724
    detector_threshold = 0.330
    context_windowsize = 5

    GB_threshold = 0.3
    GB_scoreweight = 0.3

    Detector_model_path = "C:/PHD/SideProjects/HYRAX-ID_Inference/DETECTOR_GB/ACA_26_M_SpecDyn/weights/best.pt"
    GarbageFilter_model_path = "C:/PHD/SideProjects/HYRAX-ID_Inference/DETECTOR_GB/Boost/XGBRich.joblib"
    Animal_Classifier_model_path = "C:/PHD/SideProjects/HYRAX-ID_Inference/Tdnn_Won/TEST_MHA/best_model.pt"

    audio_folder = "C:/PHD/SideProjects/VLAD/VLAD_UNPAIRED/ORIGINAL/"
    audio_denoised_folder = "C:/PHD/SideProjects/VLAD/VLAD_UNPAIRED/ACA/"
    output_path = "C:/PHD/SideProjects/VLAD/VLAD_UNPAIRED/HyraxID/"

    run_pipeline(
        Detector_model_path=Detector_model_path,
        GarbageFilter_model_path=GarbageFilter_model_path,
        Animal_Classifier_model_path=Animal_Classifier_model_path,
        audio_folder=audio_folder,
        audio_denoised_folder=audio_denoised_folder,
        output_path=output_path,
        window_size=window_size,
        hop_size=hop_size,
        image_size=image_size,
        detector_threshold=detector_threshold,
        isMulti=isMulti,
        interarrival_threshold=interarrival_threshold,
        context_windowsize=context_windowsize,
        GB_threshold=GB_threshold,
        GB_scoreweight=GB_scoreweight,
        num_classes=len(SPEAKER_LIST),
        input_dim=64,
        emb_dim=128,
    )