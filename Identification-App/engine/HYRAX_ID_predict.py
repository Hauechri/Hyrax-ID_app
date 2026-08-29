"""
End-to-end inference pipeline:

    original audio  -> Denoiser (UNet)  -> YOLO detector -> GarbageCollector (XGB) -> Bout Aggregator -> TDNN Animal Classifier -> Audacity labels
                     \\_______________________________________________________________________________/
                                            classification always uses the ORIGINAL audio

For every wav file in `audio_folder`:
  0. Denoise it with the AnimalClean UNet denoiser (same forward pass as predict.py,
     sequence_len=1s, no overlap). The denoised copy is written to `denoised_output_dir`.
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

NOTE ON THE DENOISER:
  denoise_audio_folder() mirrors predict.py exactly: model_path-only loading
  (no --checkpoint_path / --jit_load branch), --sequence_len 1, --num_workers 0,
  CUDA auto-detected, no min-max normalization, no visualization, no debug,
  and no separate log file (predict.py's Logger was dropped entirely, as
  requested -- progress just goes to stdout via print()).
"""
import os
from pathlib import Path
from collections import Counter, defaultdict
from math import ceil

import numpy as np
import joblib
import torch
import torchaudio
#YOLO stuff
import soundfile as sf
import librosa
import scipy.io.wavfile as wav
from scipy.ndimage import zoom
from ultralytics import YOLO

import matplotlib
matplotlib.use('Agg')

#Classifier Stuff
#from model_tdnn import TDNNBoutModel
from model_ECAPATDNN import ECAPATDNNBoutModel

#Denoiser Stuff.
from models.unet_model import UNet
from data.audiodataset import DefaultSpecDatasetOps, StridedAudioDataset, SingleAudioFolder
import data.transforms as T
import torch.nn as nn
from collections import OrderedDict

# =========================================================
# GLOBAL CONSTANTS
# =========================================================

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
# DENOISER (ported from predict.py)
# =========================================================

def pad_to_multiple(x, mult=16):
    """Pad a (B, C, F, T) spectrogram so F and T are both multiples of `mult`."""
    _, _, f, t = x.shape
    pad_f = (mult - f % mult) % mult
    pad_t = (mult - t % mult) % mult
    return torch.nn.functional.pad(x, (0, pad_t, 0, pad_f)), pad_f, pad_t


def load_denoiser(model_path, device):
    """
    Loads the AnimalClean UNet denoiser from a plain model_path (mirrors
    predict.py's non-jit, non-checkpoint loading branch only, since that's
    the only mode the CLI is being run in here).
    """
    model_dict = torch.load(model_path, map_location="cpu")

    # legacy key fix (same fix as predict.py)
    if "outc.conv.0.weight" in model_dict["unet"]:
        w = model_dict["unet"]["outc.conv.0.weight"]
        b = model_dict["unet"]["outc.conv.0.bias"]
        model_dict["unet"]["outc.conv.weight"] = w
        model_dict["unet"]["outc.conv.bias"] = b
        del model_dict["unet"]["outc.conv.0.weight"]
        del model_dict["unet"]["outc.conv.0.bias"]

    model = UNet(1, 1, bilinear=False)
    model.load_state_dict(model_dict["unet"])
    model = nn.Sequential(OrderedDict([("denoiser", model)]))
    dataOpts = model_dict["dataOpts"]

    model = model.to(device)
    model.eval()

    return model, dataOpts


def denoise_single_file(model, dataOpts, audio_path, sequence_len_seconds, num_workers,
                         device, debug_output_path=None):
    """
    Denoises ONE audio file entirely in memory (no intermediate wav on disk
    for the pipeline to read back) and returns (signal, sr) ready to feed
    straight into detect_elements_for_file -- i.e. peak-normalized float32
    mono, matching exactly what read_wave_file used to hand the detector
    from a written-out wav.

    If `debug_output_path` is given, the denoised audio is ALSO written to
    disk there (int16 wav) purely for inspection. The pipeline itself never
    reads that file back -- it's for you to listen to / look at, not part
    of the data flow.

    Returns (None, None) if the file produced no output (e.g. empty or
    shorter than one chunk).
    """
    sr = dataOpts["sr"]
    n_fft = dataOpts["n_fft"]
    hop_length = dataOpts["hop_length"]
    n_freq_bins = dataOpts["n_freq_bins"]
    freq_cmpr = dataOpts["freq_compression"]

    fmin = 500
    fmax = 12000

    sequence_len = int(ceil(sequence_len_seconds * sr))
    hop = sequence_len  # no overlap between chunks, same as predict.py

    t_decompr_f = T.Decompress(
        f_min=fmin,
        f_max=fmax,
        n_fft=n_fft,
        sr=sr
    ).to(device)
    window = torch.hann_window(n_fft)
    if device.type == "cuda":
        window = window.cuda()

    dataset = StridedAudioDataset(
        audio_path,
        sequence_len=sequence_len,
        hop=hop,
        sr=sr,
        fft_size=n_fft,
        fft_hop=hop_length,
        n_freq_bins=n_freq_bins,
        f_min=fmin,
        f_max=fmax,
        freq_compression=freq_cmpr,
        center=True,
        min_max_normalize=False,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_audio = None

    with torch.no_grad():
        for batch in data_loader:
            _, input_spec, spec_cmplx, _ = batch

            if device.type == "cuda":
                input_spec = input_spec.cuda()
                spec_cmplx = spec_cmplx.cuda()

            input_spec, pad_f, pad_t = pad_to_multiple(input_spec, mult=16)
            denoised = model(input_spec)

            if pad_f > 0:
                denoised = denoised[:, :, :-pad_f, :]
            if pad_t > 0:
                denoised = denoised[:, :, :, :-pad_t]

            decompressed = t_decompr_f(denoised)

            spec_cmplx = spec_cmplx.squeeze(0)
            decompressed = decompressed.unsqueeze(-1)

            audio_spec = decompressed * spec_cmplx
            audio_spec = audio_spec.squeeze(0).transpose(0, 1)
            audio_spec = torch.view_as_complex(audio_spec)

            audio_chunk = torch.istft(
                audio_spec,
                n_fft,
                hop_length=hop_length,
                onesided=True,
                center=True,
                window=window,
            )

            total_audio = audio_chunk if total_audio is None else torch.cat((total_audio, audio_chunk), dim=0)

            del denoised, decompressed, audio_spec, audio_chunk
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if total_audio is None:
        return None, None

    total_audio_np = total_audio.cpu().numpy().astype(np.float32)

    if debug_output_path is not None:
        os.makedirs(os.path.dirname(debug_output_path), exist_ok=True)
        debug_int16 = (total_audio_np * np.iinfo(np.int16).max).astype(np.int16)
        wav.write(debug_output_path, sr, debug_int16)

    # Match read_wave_file's normalization exactly (peak -> [-1, 1], mono)
    # so detect_elements_for_file sees the same kind of input it always has.
    if total_audio_np.ndim > 1:
        total_audio_np = total_audio_np.mean(axis=1)
    peak = np.max(np.abs(total_audio_np))
    if peak > 0:
        total_audio_np = total_audio_np / peak

    return total_audio_np, sr


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
    signal,
    sr,
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
    single, already-denoised, in-memory signal (peak-normalized float32
    mono, as returned by denoise_single_file).

    Returns:
        elements (list of (start, end, cls_name, conf)) : surviving elements
        bouts    (list of (start, end, "BOUT", bout_number))
    """
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
    Denoiser_model_path,
    Denoiser_sequence_length,
    Denoiser_num_worker,
    audio_folder,
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
    emb_dim=192,
    debug_denoised_dir=None,
    device_override=None,
    cancel_event=None,
):
    """
    debug_denoised_dir: if set, every denoised file is ALSO written here as
    a wav for inspection. Leave as None for the clean, no-tmp-files final
    run -- the pipeline never needs these files itself.

    device_override: "cpu", "cuda", or None/"auto" to keep the original
    auto-detect behavior. Lets a caller (e.g. a GUI device selector) force
    a specific device without relying on process-global env vars.

    cancel_event: an optional threading.Event; checked once per file so a
    caller can request a clean stop between files (not mid-file).
    """
    if device_override in (None, "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_override)
    print(f"[INFO] Using device: {device}")

    denoiser_model, dataOpts = load_denoiser(Denoiser_model_path, device)
    print(f"[INFO] Denoiser loaded: {Denoiser_model_path}")

    detector = YOLO(Detector_model_path)
    print(f"[INFO] Detector loaded: {Detector_model_path}")

    gc_data = joblib.load(GarbageFilter_model_path)
    garbagecollector = gc_data["model"]
    trained_feature_keys = list(gc_data["feature_keys"])
    print(f"[INFO] GarbageCollector loaded: {GarbageFilter_model_path}")

    #classifier = TDNNBoutModel(num_classes=num_classes, input_dim=input_dim, emb_dim=emb_dim)
    classifier = ECAPATDNNBoutModel(num_classes=num_classes, input_dim=input_dim, emb_dim=emb_dim)

    classifier.load_state_dict(torch.load(Animal_Classifier_model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    print(f"[INFO] Animal classifier loaded: {Animal_Classifier_model_path}")

    os.makedirs(output_path, exist_ok=True)
    if debug_denoised_dir is not None:
        os.makedirs(debug_denoised_dir, exist_ok=True)

    originals = sorted(collect_wav_files(audio_folder).items())
    print(f"[INFO] Found {len(originals)} audio file(s) to process.")

    for file_index, (stem, original_path) in enumerate(originals):
        if cancel_event is not None and cancel_event.is_set():
            print("[INFO] Cancelled.")
            break

        print(f"\n[FILE {file_index + 1}/{len(originals)}] Processing: {stem}")

        debug_path = os.path.join(debug_denoised_dir, f"{stem}.wav") if debug_denoised_dir else None

        denoised_signal, denoised_sr = denoise_single_file(
            denoiser_model,
            dataOpts,
            original_path,
            Denoiser_sequence_length,
            Denoiser_num_worker,
            device,
            debug_output_path=debug_path,
        )

        if denoised_signal is None:
            print(f"  [WARN] {stem}: denoiser produced no output (empty/too-short file?), skipping.")
            continue

        elements, bouts = detect_elements_for_file(
            detector,
            garbagecollector,
            trained_feature_keys,
            denoised_signal,
            denoised_sr,
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

    Denoiser_sequence_length = 1
    Denoiser_num_worker = 0

    Detector_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/DETECTOR_GB/ACA_26_M_SpecDyn/weights/best.pt"
    GarbageFilter_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/DETECTOR_GB/Boost/XGBRich.joblib"
    #Animal_Classifier_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/TDNN_MHA/All_MHA/best_model.pt"
    Animal_Classifier_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/ECAPA_TDNN/Final_ECAPA_TDNN_ATTENTIVE_ARCFACE/best_model.pt"

    Denoiser_model_path = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/01_ACS.pk"

    audio_folder = "C:/PHD/SideProjectsData/HeloiseNewDownload/male_songs_combined_paired/Audio/"
    output_path = "C:/PHD/SideProjectsData/HeloiseNewDownload/male_songs_combined_paired/HyraxID_ECAPA/"
    debug_denoised_dir = None  # os.path.join(output_path, "denoised_tmp")

    # Denoised wavs are now generated by the pipeline itself instead of
    # being a required pre-existing input folder.


    run_pipeline(
        Detector_model_path=Detector_model_path,
        GarbageFilter_model_path=GarbageFilter_model_path,
        Animal_Classifier_model_path=Animal_Classifier_model_path,
        Denoiser_model_path=Denoiser_model_path,
        Denoiser_sequence_length=Denoiser_sequence_length,
        Denoiser_num_worker=Denoiser_num_worker,
        audio_folder=audio_folder,
        debug_denoised_dir=debug_denoised_dir,
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
        emb_dim=192
    )