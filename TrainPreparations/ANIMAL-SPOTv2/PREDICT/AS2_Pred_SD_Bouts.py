import os
from pathlib import Path
from ultralytics import YOLO
import librosa
import scipy.io.wavfile as wav
import numpy as np
from scipy.ndimage import zoom

import matplotlib
matplotlib.use('Agg')



def collect_wav_files(folder_path):
    """Collect all .wav files from the specified folder."""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.wav')]


def read_wave_file(filepath):
    """Read a wave file and normalize the signal."""
    rate, data = wav.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))  # Normalize to [-1, 1]
    return data, rate


def slice_audio_signal(signal, window_size, hop_size):
    """Slice audio signal into overlapping windows."""
    total_samples = len(signal)
    slices = []

    for start in range(0, total_samples - window_size + 1, hop_size):
        end = start + window_size
        slices.append((signal[start:end], start, end))

    return slices


def prepare_numpy_for_yolo(numpy_array):
    img = np.clip(numpy_array, numpy_array.min(), numpy_array.max())
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    img_rgb = np.dstack([img] * 3)
    return img_rgb

def run_inference(model, numpy_array, threshold=0.05):
    #img = prepare_numpy_for_yolo(numpy_array)
    results = model(numpy_array, conf=threshold)
    return results[0]


def convert_boxes_to_audacity_labels(prediction, image_duration_sec, start_sample, sample_rate, image_width, model=None,
                                     isMulti=False):
    """Convert YOLO bounding boxes to Audacity labels (start_sec, end_sec, class_name)."""
    labels = []

    # Fallback mapping if model names are not available
    class_map = {0: "chuck", 1: "snort", 2: "wail"} if isMulti else {0: "event"}

    for box in prediction.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box

        # Convert x-coordinates to time within the slice
        box_start_time = (x1 / image_width) * image_duration_sec
        box_end_time = (x2 / image_width) * image_duration_sec

        # Adjust by the slice offset
        global_start_time = (start_sample / sample_rate) + box_start_time
        global_end_time = (start_sample / sample_rate) + box_end_time

        # Get label name
        if model is not None and hasattr(model, "names") and int(cls) in model.names:
            cls_name = str(model.names[int(cls)]).lower()
        else:
            cls_name = class_map.get(int(cls), f"class_{int(cls)}")

        labels.append((global_start_time, global_end_time, cls_name, float(conf)))

    return labels


def save_audacity_labels(label_data, output_path):
    """Save label list to a file in Audacity label format."""
    with open(output_path, 'w') as f:
        for start, end, cls_name, conf in label_data:
            f.write(f"{start:.6f}\t{end:.6f}\t{cls_name}_{conf:.6f}\n")

def merge_overlapping_labels(label_list, overlap_threshold=0.5, min_fraction_keep=0.5):
    """
    Merge labels of the same class if they overlap by at least a given fraction,
    then resolve overlaps between different classes based on confidence.

    Args:
        label_list (List[Tuple[float, float, int|str, float]]):
            (start, end, class_id, confidence)
        overlap_threshold (float):
            Minimum fractional overlap (0–1) to merge two intervals.
        min_fraction_keep (float):
            Minimum fraction of original duration to keep after trimming.

    Returns:
        List[Tuple[float, float, int|str, float]]: Merged and non-overlapping labels.
    """
    from collections import defaultdict

    def overlap_fraction(a_start, a_end, b_start, b_end):
        """Return fractional overlap relative to the smaller interval."""
        overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
        len_a, len_b = a_end - a_start, b_end - b_start
        denom = min(len_a, len_b)
        return overlap / denom if denom > 0 else 0

    # --- STEP 1: Merge overlapping labels of the same class ---
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

    # --- STEP 2: Resolve overlaps between *different classes* ---
    final_labels = []
    merged_labels.sort(key=lambda x: x[0])

    for label in merged_labels:
        start, end, cls, conf = label
        keep = True
        for i, (s2, e2, c2, conf2) in enumerate(final_labels):
            # Check overlap
            if e2 <= start or s2 >= end:
                continue  # no overlap

            overlap_start = max(start, s2)
            overlap_end = min(end, e2)
            overlap_len = overlap_end - overlap_start

            if overlap_len <= 0:
                continue

            # Decide who wins
            if conf > conf2:
                # Current label wins — modify the existing one
                orig_len = e2 - s2
                if overlap_start <= s2 and overlap_end >= e2:
                    # Fully covered — remove
                    final_labels[i] = None
                elif overlap_start <= s2:
                    # Trim start
                    new_start = overlap_end
                    new_len = e2 - new_start
                    if new_len >= min_fraction_keep * orig_len:
                        final_labels[i] = (new_start, e2, c2, conf2)
                    else:
                        final_labels[i] = None
                elif overlap_end >= e2:
                    # Trim end
                    new_end = overlap_start
                    new_len = new_end - s2
                    if new_len >= min_fraction_keep * orig_len:
                        final_labels[i] = (s2, new_end, c2, conf2)
                    else:
                        final_labels[i] = None
                else:
                    # Overlap in the middle → split? too complex, remove weaker
                    final_labels[i] = None
            else:
                # Existing label wins — modify or remove current one
                orig_len = end - start
                if overlap_start <= start and overlap_end >= end:
                    # Fully covered
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
                    # Overlap in middle → weaker one loses completely
                    keep = False
                    break

        # Clean up None entries
        final_labels = [lbl for lbl in final_labels if lbl is not None]

        if keep:
            final_labels.append((start, end, cls, conf))

    # Final sort
    final_labels.sort(key=lambda x: x[0])
    return final_labels

def group_bouts(merged_labels, interarrival_threshold=1.0):
    """
    Group merged labels into bouts based on interarrival time.

    Args:
        merged_labels (List[Tuple[float, float, str, float]]): Merged labels
        interarrival_threshold (float): Max time gap to consider same bout

    Returns:
        List[Tuple[float, float, str, float]]: Bout labels as (start, end, 'BOUT', bout_number)
    """
    if not merged_labels:
        return []

    # Sort labels by start time
    merged_labels.sort(key=lambda x: x[0])

    bouts = []
    bout_start = merged_labels[0][0]
    bout_end = merged_labels[0][1]
    bout_number = 1

    for i in range(1, len(merged_labels)):
        start, end, cls, conf = merged_labels[i]
        # Check interarrival time between this label and previous
        gap = start - bout_end
        if gap <= interarrival_threshold:
            # Same bout → extend end
            bout_end = max(bout_end, end)
        else:
            # New bout → save previous
            bouts.append((bout_start, bout_end, "BOUT", float(bout_number)))
            bout_number += 1
            bout_start = start
            bout_end = end

    # Add last bout
    bouts.append((bout_start, bout_end, "BOUT", float(bout_number)))
    return bouts


def main(model_path, audio_folder, output_path, window_size, hop_size, image_size, threshold, isMulti, interarrival_threshold):
    model = YOLO(model_path)
    print(f"[INFO] Model loaded: {model_path}")

    audio_files = collect_wav_files(audio_folder)
    print(f"[INFO] Found {len(audio_files)} audio files.")

    all_labels = []

    for file_index, file_path in enumerate(audio_files):
        print(f"\n[FILE {file_index + 1}/{len(audio_files)}] Processing: {Path(file_path).name}")
        signal, sr = read_wave_file(file_path)
        slices = slice_audio_signal(signal, window_size, hop_size)

        for slice_index, (audio_slice, start, _) in enumerate(slices):
            print(f"  - Slice {slice_index + 1}/{len(slices)}", end='\r')
            input_img = generate_input_image(audio_slice, sr, (image_size, image_size))
            prediction = run_inference(model, input_img, threshold=threshold)
            labels = convert_boxes_to_audacity_labels(
                prediction,
                image_duration_sec=window_size / sr,
                start_sample=start,
                sample_rate=sr,
                image_width=image_size,
                model=model,
                isMulti=isMulti
            )
            all_labels.extend(labels)

        # Save labels per file
        label_Filename = os.path.splitext(os.path.basename(file_path))[0] + '_labels.txt'
        label_output_path = os.path.join(output_path, label_Filename)

        merged_labels = merge_overlapping_labels(all_labels)
        bout_labels = group_bouts(merged_labels, interarrival_threshold=interarrival_threshold)
        combined_labels = merged_labels + bout_labels
        combined_labels.sort(key=lambda x: x[0])

        save_audacity_labels(combined_labels, label_output_path)
        print(f"\n  ✓ Labels saved to {label_output_path}")

        # Reset label collection for next file
        all_labels.clear()


def resize(input, image_size=(800, 800)):
    """Resize spectrogram to match model input size."""
    target_h, target_w = image_size
    src_h, src_w = input.shape
    zoom_factors = (target_h / src_h, target_w / src_w)
    return zoom(input, zoom=zoom_factors, order=3)


def SpectroDynamics(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 60, width: int = 9):
    """
    Compute SpectroDynamics: base, delta, and delta-delta spectrograms.
    Returns:
        spec0: normalized log power spectrogram
        spec1: first-order delta (Δ)
        spec2: second-order delta (ΔΔ)
    """

    # === Base STFT and log power ===
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    S_db = librosa.power_to_db(S, ref=np.max)

    # === Deltas ===
    delta  = librosa.feature.delta(S_db, width=width, order=1)
    delta2 = librosa.feature.delta(S_db, width=width, order=2)

    # === Normalize each to [0, 1] ===
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    spec0 = norm(S_db)
    spec1 = norm(delta)
    spec2 = norm(delta2)

    return spec0, spec1, spec2

def generate_input_image(y, sr, image_size):
    """Convert audio slice into spectrogram image."""
    spec0, spec1, spec2 = SpectroDynamics(y,sr)
    # --- Spectrogram ---
    spec0 = resize(spec0, image_size)
    # --- ΔSpectrogram ---
    spec1 = resize(spec1, image_size)
    # --- ΔΔSpectrogram ---
    spec2 = resize(spec2, image_size)

    R = np.clip(spec0, spec0.min(), spec0.max())
    R = ((R - R.min()) / (R.max() - R.min()) * 255).astype(np.uint8)

    G = np.clip(spec1, spec1.min(), spec1.max())
    G = ((G - G.min()) / (G.max() - G.min()) * 255).astype(np.uint8)

    B = np.clip(spec2, spec2.min(), spec2.max())
    B = ((B - B.min()) / (B.max() - B.min()) * 255).astype(np.uint8)


    # 2. Stack into 3 channels (RGB)
    img_rgb = np.dstack([R, G, B])
    return img_rgb


if __name__ == "__main__":
    # === SETTINGS ===
    window_size = 48000
    hop_size = 24000
    image_size = 800
    isMulti = True
    interarrival_threshold = 0.724
#OLD #0.30 BioAdapt#0.60 on Noise 0.40 on Denoised# 0.35 on AC and ACA
    #ORG 0.370
    #BIOD 0.430
    #BIODA 0.360
    #ACA 0.330
    #AC 0.590
    threshold = 0.330

    model_path = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/Models/ACA_26_M_ICASSP/weights/best.pt"
    audio_folder = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/ACA/"
    output_path = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/Predicts/ACA/"
    main(model_path, audio_folder, output_path, window_size, hop_size, image_size, threshold, isMulti, interarrival_threshold)


