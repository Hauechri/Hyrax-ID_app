import os
import re
import numpy as np
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

INPUT_FOLDER = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/BoostTrain/Both"
OUTPUT_FOLDER = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/BoostTrain/FIT/"

WINDOW_SIZE = 5

LABEL_MAP = {
    "chuck": 0,
    "snort": 1,
    "wail": 2,
}

FILENAME_REGEX = re.compile(
    r"(?P<base>.+)___(?P<tag>TP|FP|MS)_(?P<idx>\d+)_start(?P<start>\d+\.\d+)_end(?P<end>\d+\.\d+)_(?P<label>\w+)_conf(?P<conf>\d+\.\d+)"
)

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
            "duration": float(d["end"]) - float(d["start"]),
            "conf": float(d["conf"]),
            "class_id": LABEL_MAP[d["label"]],
            "idx": int(d["idx"]),
            "tag": d["tag"],
        }

        bases.setdefault(base, []).append(det)

    # Sort by temporal order
    for base in bases:
        bases[base].sort(key=lambda x: x["start"])

    print(f"Loaded {len(bases)} recordings.")

    return bases


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def compute_rich_features(detections, window=5):

    N = len(detections)

    starts = np.array([d["start"] for d in detections], dtype=np.float32)
    ends = np.array([d["end"] for d in detections], dtype=np.float32)
    durations = np.array([d["duration"] for d in detections], dtype=np.float32)
    confs = np.array([d["conf"] for d in detections], dtype=np.float32)
    classes = np.array([d["class_id"] for d in detections], dtype=np.float32)

    X = []
    y = []
    meta = []

    for i in range(N):

        current_start = starts[i]

        feat = {}

        # =====================================================
        # CENTRAL SAMPLE
        # =====================================================

        feat["center_conf"] = confs[i]
        feat["center_duration"] = durations[i]
        feat["center_class"] = classes[i]

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

        # =====================================================
        # WINDOW SUMMARY FEATURES
        # =====================================================

        feat["window_conf_mean"] = nanmean(neighbor_confs)
        feat["window_conf_std"] = nanstd(neighbor_confs)

        feat["window_duration_mean"] = nanmean(neighbor_durations)
        feat["window_duration_std"] = nanstd(neighbor_durations)

        feat["window_gap_mean"] = nanmean(neighbor_gaps)
        feat["window_gap_std"] = nanstd(neighbor_gaps)

        feat["window_density"] = len(neighbor_gaps)

        # =====================================================
        # RANK / RELATIVE FEATURES
        # =====================================================

        all_confs = np.array(neighbor_confs + [confs[i]])

        conf_rank = np.sum(all_confs < confs[i])

        feat["conf_rank"] = conf_rank
        feat["conf_percentile"] = conf_rank / len(all_confs)

        feat["is_local_max_conf"] = (
            1.0 if confs[i] >= np.nanmax(all_confs) else 0.0
        )

        feat["is_local_min_conf"] = (
            1.0 if confs[i] <= np.nanmin(all_confs) else 0.0
        )

        # =====================================================
        # CLASS CONTEXT FEATURES
        # =====================================================

        if len(local_classes) > 0:

            class_counts = Counter(local_classes)

            majority_class = class_counts.most_common(1)[0][0]

            feat["same_class_ratio"] = (
                np.sum(np.array(local_classes) == classes[i])
                / len(local_classes)
            )

            feat["is_majority_class"] = (
                1.0 if classes[i] == majority_class else 0.0
            )

        else:

            feat["same_class_ratio"] = np.nan
            feat["is_majority_class"] = np.nan

        # =====================================================
        # LABEL
        # =====================================================

        label = 1 if detections[i]["tag"] in ("TP", "MS") else 0

        # =====================================================
        # SAVE
        # =====================================================

        X.append(feat)
        y.append(label)

        meta.append({
            "file": detections[i]["file"],
            "idx": detections[i]["idx"],
            "start": detections[i]["start"],
            "end": detections[i]["end"],
            "tag": detections[i]["tag"],
        })

    return X, np.array(y, dtype=np.int64), meta


# =========================================================
# FEATURE MATRIX CONVERSION
# =========================================================

def features_to_matrix(feature_list):

    keys = sorted(feature_list[0].keys())

    X = []

    for feat in feature_list:
        X.append([feat[k] for k in keys])

    X = np.array(X, dtype=np.float32)

    return X, keys


# =========================================================
# SAVE PER RECORDING
# =========================================================

def save_recording(base, X, y, meta, feature_keys, out_folder):

    out_path = os.path.join(out_folder, f"{base}.npz")

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        meta=np.array(meta, dtype=object),
        feature_keys=np.array(feature_keys, dtype=object),
    )

    print(f"Saved: {base}.npz ({len(y)} samples)")


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Loading detections...")
    bases = load_detections(INPUT_FOLDER)

    print("\n==============================")
    print(" RECORDING SAMPLE COUNTS ")
    print("==============================")

    total_samples = 0

    for base, detections in bases.items():

        num_samples = len(detections)
        total_samples += num_samples

        print(f"{base}: {num_samples} samples")

        feats, y, meta = compute_rich_features(
            detections,
            window=WINDOW_SIZE
        )

        X, feature_keys = features_to_matrix(feats)

        save_recording(
            base,
            X,
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