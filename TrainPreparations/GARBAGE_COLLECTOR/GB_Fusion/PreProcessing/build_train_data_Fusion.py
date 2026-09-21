# =========================================================
# IMPORTS
# =========================================================

import os
import re
import numpy as np

import soundfile as sf

import torch
import torchaudio

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

from collections import Counter


# =========================================================
# CONFIG
# =========================================================

INPUT_FOLDER = r"D:/0_PHD/Hyrax/BoutGarbageCollector/DATA/ACA/GoodTargetGarbage/Both/"
OUTPUT_FOLDER = r"D:/0_PHD/Hyrax/BoutGarbageCollector/DATA/ACA/GoodTargetGarbage/Features_Fusion/"

WINDOW_SIZE = 5

# audio padding around event
EMBED_PADDING = 0.25

MODEL_NAME = "facebook/hubert-base-ls960"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

LABEL_MAP = {
    "chuck": 0,
    "snort": 1,
    "wail": 2,
}

FILENAME_REGEX = re.compile(
    r"(?P<base>.+)___(?P<tag>TP|FP|MS)_(?P<idx>\d+)_start(?P<start>\d+\.\d+)_end(?P<end>\d+\.\d+)_(?P<label>\w+)_conf(?P<conf>\d+\.\d+)"
)


# =========================================================
# LOAD HUBERT
# =========================================================

print("\nLoading HuBERT...")

processor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_NAME
)

hubert = HubertModel.from_pretrained(
    MODEL_NAME
)

hubert.to(DEVICE)
hubert.eval()

print(f"HuBERT loaded on {DEVICE}")


# =========================================================
# HELPERS
# =========================================================

def safe_get(arr, idx, default=np.nan):

    if idx < 0 or idx >= len(arr):
        return default

    return arr[idx]


def nanmean(x):
    return np.nan if len(x) == 0 else np.nanmean(x)


def nanstd(x):
    return np.nan if len(x) == 0 else np.nanstd(x)


# =========================================================
# HUBERT EMBEDDING EXTRACTION
# =========================================================

def extract_hubert_embedding(
    audio,
    sr,
):

    # mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # float32
    audio = audio.astype(np.float32)

    # resample to 16k
    if sr != 16000:

        resampler = torchaudio.transforms.Resample(
            sr,
            16000
        )

        audio_tensor = torch.tensor(audio)

        audio_tensor = resampler(
            audio_tensor
        )

        audio = audio_tensor.numpy()

        sr = 16000

    # tokenize
    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    input_values = (
        inputs.input_values
        .to(DEVICE)
    )

    # inference
    with torch.no_grad():

        outputs = hubert(
            input_values
        )

        hidden = (
            outputs.last_hidden_state
        )

        # mean pooling
        embedding = hidden.mean(dim=1)

    embedding = (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # L2 normalize
    embedding = embedding / (
        np.linalg.norm(embedding)
        + 1e-9
    )

    return embedding.astype(np.float32)


# =========================================================
# LOAD DETECTIONS
# =========================================================

def load_detections(folder):

    bases = {}

    for fname in os.listdir(folder):

        if not fname.endswith(".wav"):
            continue

        match = FILENAME_REGEX.match(fname)

        if not match:
            print(f"Skipping: {fname}")
            continue

        d = match.groupdict()

        base = d["base"]

        det = {
            "file": fname,
            "start": float(d["start"]),
            "end": float(d["end"]),
            "duration": (
                float(d["end"])
                - float(d["start"])
            ),
            "conf": float(d["conf"]),
            "class_id": LABEL_MAP[d["label"]],
            "idx": int(d["idx"]),
            "tag": d["tag"],
        }

        bases.setdefault(base, []).append(det)

    # sort temporally
    for base in bases:
        bases[base].sort(
            key=lambda x: x["start"]
        )

    print(f"Loaded {len(bases)} recordings.")

    return bases


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def compute_rich_features(
    detections,
    waveform,
    sr,
    window=5
):

    N = len(detections)

    starts = np.array(
        [d["start"] for d in detections],
        dtype=np.float32
    )

    ends = np.array(
        [d["end"] for d in detections],
        dtype=np.float32
    )

    durations = np.array(
        [d["duration"] for d in detections],
        dtype=np.float32
    )

    confs = np.array(
        [d["conf"] for d in detections],
        dtype=np.float32
    )

    classes = np.array(
        [d["class_id"] for d in detections],
        dtype=np.float32
    )

    X = []
    y = []
    meta = []

    for i in range(N):

        print(
            f"    sample {i+1}/{N}",
            end="\r"
        )

        current_start = starts[i]

        feat = {}

        # =====================================================
        # CENTRAL SAMPLE
        # =====================================================

        feat["center_conf"] = confs[i]
        feat["center_duration"] = durations[i]
        feat["center_class"] = classes[i]

        # =====================================================
        # HUBERT EMBEDDING
        # =====================================================

        clip = waveform

        embedding = extract_hubert_embedding(
            clip,
            sr
        )

        feat["embedding"] = embedding

        # =====================================================
        # NEIGHBOR FEATURES
        # =====================================================

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

                dt = (
                    starts[j]
                    - current_start
                )

                feat[f"{prefix}_exists"] = 1.0
                feat[f"{prefix}_dt"] = dt
                feat[f"{prefix}_conf"] = confs[j]
                feat[f"{prefix}_duration"] = durations[j]
                feat[f"{prefix}_class"] = classes[j]

                feat[f"{prefix}_delta_conf"] = (
                    confs[j]
                    - confs[i]
                )

                neighbor_confs.append(
                    confs[j]
                )

                neighbor_durations.append(
                    durations[j]
                )

                neighbor_gaps.append(
                    abs(dt)
                )

                local_classes.append(
                    classes[j]
                )

            else:

                feat[f"{prefix}_exists"] = 0.0
                feat[f"{prefix}_dt"] = np.nan
                feat[f"{prefix}_conf"] = np.nan
                feat[f"{prefix}_duration"] = np.nan
                feat[f"{prefix}_class"] = np.nan
                feat[f"{prefix}_delta_conf"] = np.nan

        # =====================================================
        # WINDOW FEATURES
        # =====================================================

        feat["window_conf_mean"] = (
            nanmean(neighbor_confs)
        )

        feat["window_conf_std"] = (
            nanstd(neighbor_confs)
        )

        feat["window_duration_mean"] = (
            nanmean(neighbor_durations)
        )

        feat["window_duration_std"] = (
            nanstd(neighbor_durations)
        )

        feat["window_gap_mean"] = (
            nanmean(neighbor_gaps)
        )

        feat["window_gap_std"] = (
            nanstd(neighbor_gaps)
        )

        feat["window_density"] = (
            len(neighbor_gaps)
        )

        # =====================================================
        # RANK FEATURES
        # =====================================================

        all_confs = np.array(
            neighbor_confs + [confs[i]]
        )

        conf_rank = np.sum(
            all_confs < confs[i]
        )

        feat["conf_rank"] = conf_rank

        feat["conf_percentile"] = (
            conf_rank / len(all_confs)
        )

        feat["is_local_max_conf"] = (
            1.0
            if confs[i] >= np.nanmax(all_confs)
            else 0.0
        )

        feat["is_local_min_conf"] = (
            1.0
            if confs[i] <= np.nanmin(all_confs)
            else 0.0
        )

        # =====================================================
        # CLASS FEATURES
        # =====================================================

        if len(local_classes) > 0:

            class_counts = Counter(
                local_classes
            )

            majority_class = (
                class_counts
                .most_common(1)[0][0]
            )

            feat["same_class_ratio"] = (
                np.sum(
                    np.array(local_classes)
                    == classes[i]
                )
                / len(local_classes)
            )

            feat["is_majority_class"] = (
                1.0
                if classes[i] == majority_class
                else 0.0
            )

        else:

            feat["same_class_ratio"] = np.nan
            feat["is_majority_class"] = np.nan

        # =====================================================
        # LABEL
        # =====================================================

        label = (
            1
            if detections[i]["tag"]
               in ("TP", "MS")
            else 0
        )

        X.append(feat)
        y.append(label)

        meta.append({
            "file": detections[i]["file"],
            "idx": detections[i]["idx"],
            "start": detections[i]["start"],
            "end": detections[i]["end"],
            "tag": detections[i]["tag"],
        })

    print()

    return (
        X,
        np.array(y, dtype=np.int64),
        meta
    )


# =========================================================
# MATRIX CONVERSION
# =========================================================

def features_to_matrix(
    feature_list
):

    feature_keys = sorted([
        k for k in feature_list[0].keys()
        if k != "embedding"
    ])

    X_context = []
    X_embed = []

    for feat in feature_list:

        X_context.append([
            feat[k]
            for k in feature_keys
        ])

        X_embed.append(
            feat["embedding"]
        )

    X_context = np.array(
        X_context,
        dtype=np.float32
    )

    X_embed = np.array(
        X_embed,
        dtype=np.float32
    )

    return (
        X_context,
        X_embed,
        feature_keys
    )


# =========================================================
# SAVE
# =========================================================

def save_recording(
    base,
    X_context,
    X_embed,
    y,
    meta,
    feature_keys,
    out_folder
):

    out_path = os.path.join(
        out_folder,
        f"{base}.npz"
    )

    np.savez_compressed(
        out_path,

        X_context=X_context,
        X_embed=X_embed,

        y=y,

        meta=np.array(
            meta,
            dtype=object
        ),

        feature_keys=np.array(
            feature_keys,
            dtype=object
        ),
    )

    print(
        f"Saved: {base}.npz "
        f"({len(y)} samples)"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    print("Loading detections...")

    bases = load_detections(
        INPUT_FOLDER
    )

    total_samples = 0

    for base, detections in bases.items():

        print("\n==============================")
        print(f"Processing: {base}")
        print("==============================")

        total_samples += len(detections)

        # =====================================================
        # LOAD FULL RECORDING
        # =====================================================

        waveform, sr = sf.read(
            os.path.join(
                INPUT_FOLDER,
                detections[0]["file"]
            )
        )

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        waveform = waveform.astype(
            np.float32
        )

        # =====================================================
        # FEATURES
        # =====================================================

        feats, y, meta = (
            compute_rich_features(
                detections,
                waveform,
                sr,
                window=WINDOW_SIZE
            )
        )

        (
            X_context,
            X_embed,
            feature_keys
        ) = features_to_matrix(
            feats
        )

        # =====================================================
        # SAVE
        # =====================================================

        save_recording(
            base,
            X_context,
            X_embed,
            y,
            meta,
            feature_keys,
            OUTPUT_FOLDER
        )

    print("\n==============================")
    print(f"Total recordings: {len(bases)}")
    print(f"Total samples: {total_samples}")
    print("==============================")

    print("Done.")


if __name__ == "__main__":
    main()