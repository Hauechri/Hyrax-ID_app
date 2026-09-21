import os
import re
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


# =========================
# CONFIG
# =========================

LABEL_MAP = {
    "chuck": 0,
    "snort": 1,
    "wail": 2,
}

FEATURE_KEYS = [
    "conf",
    "duration",
    "class_id",
    "dt_prev",
    "dt_next",
    "conf_prev1",
    "conf_prev2",
    "conf_next1",
    "conf_next2",
    "dt_prev2",
    "dt_next2",
]

FILENAME_REGEX = re.compile(
    r"(?P<base>.+)___(?P<tag>TP|FP|MS)_(?P<idx>\d+)_start(?P<start>\d+\.\d+)_end(?P<end>\d+\.\d+)_(?P<label>\w+)_conf(?P<conf>\d+\.\d+)(?:_iou(?P<iou>\d+\.\d+))?"
)


# =========================
# FEATURE ENGINEERING
# =========================

def safe_get(arr, idx, default=0.0):
    if idx < 0 or idx >= len(arr):
        return default
    return arr[idx]


def compute_features(detections):
    N = len(detections)

    starts = np.array([d["start"] for d in detections])
    ends = np.array([d["end"] for d in detections])
    confs = np.array([d["conf"] for d in detections])
    classes = np.array([d["class_id"] for d in detections])
    durations = ends - starts

    features = []

    for i in range(N):
        dt_prev = starts[i] - ends[i - 1] if i > 0 else 999.0
        dt_next = starts[i + 1] - ends[i] if i < N - 1 else 999.0

        feat = {
            "conf": confs[i],
            "duration": durations[i],
            "class_id": classes[i],

            "dt_prev": dt_prev,
            "dt_next": dt_next,

            "conf_prev1": safe_get(confs, i - 1),
            "conf_prev2": safe_get(confs, i - 2),
            "conf_next1": safe_get(confs, i + 1),
            "conf_next2": safe_get(confs, i + 2),

            "dt_prev2": starts[i] - ends[i - 2] if i > 1 else 999.0,
            "dt_next2": starts[i + 2] - ends[i] if i < N - 2 else 999.0,
        }

        features.append(feat)

    return features


def features_to_matrix(feature_list):
    X = []
    for f in feature_list:
        X.append([f[k] for k in FEATURE_KEYS])
    return np.array(X, dtype=np.float32)


# =========================
# DATA LOADING
# =========================

def load_detections(folder):
    bases = {}

    for fname in os.listdir(folder):
        if not fname.endswith(".wav"):
            continue

        match = FILENAME_REGEX.match(fname)
        if not match:
            print(f"Skipping (no match): {fname}")
            continue

        d = match.groupdict()

        base = d["base"]

        det = {
            "start": float(d["start"]),
            "end": float(d["end"]),
            "conf": float(d["conf"]),
            "class_id": LABEL_MAP[d["label"]],
            "idx": int(d["idx"]),
            "tag": d["tag"],  # TP, FP, MS
        }

        bases.setdefault(base, []).append(det)

    # sort detections within each base
    for base in bases:
        bases[base].sort(key=lambda x: x["idx"])

    print(f"Loaded {len(bases)} bases.")
    return bases


# =========================
# BUILD DATASET
# =========================

def build_features_and_labels(bases):
    X_all = []
    y_all = []
    groups = []

    for base, detections in bases.items():
        feats = compute_features(detections)

        for f, det in zip(feats, detections):
            label = 1 if det["tag"] in ("TP", "MS") else 0

            X_all.append(f)
            y_all.append(label)
            groups.append(base)

    return X_all, np.array(y_all), np.array(groups)


# =========================
# SPLITTING
# =========================

def split_data(X, y, groups):
    X = np.array(X, dtype=object)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(X, y, groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(
        gss2.split(X[temp_idx], y[temp_idx], groups[temp_idx])
    )

    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    return train_idx, val_idx, test_idx


# =========================
# SAVE
# =========================

def save_split(name, indices, X, y, out_folder):
    X_sel = [X[i] for i in indices]
    y_sel = y[indices]

    X_mat = features_to_matrix(X_sel)

    np.savez(
        os.path.join(out_folder, f"{name}.npz"),
        X=X_mat,
        y=y_sel
    )

    print(f"{name}: {len(indices)} samples saved.")


# =========================
# MAIN
# =========================

def main(data_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    print("Loading detections...")
    bases = load_detections(data_folder)

    print("Building features...")
    X, y, groups = build_features_and_labels(bases)

    print(f"Total samples: {len(y)}")
    print(f"KEEP ratio: {np.mean(y):.3f}")

    print("Splitting...")
    train_idx, val_idx, test_idx = split_data(X, y, groups)

    save_split("train", train_idx, X, y, out_folder)
    save_split("val", val_idx, X, y, out_folder)
    save_split("test", test_idx, X, y, out_folder)

    print("Done.")


if __name__ == "__main__":
    Data_folder = "D:/0_PHD/Hyrax/BoutGarbageCollector/AUDIO/ACA/GoodTargetGarbage/Both/"
    Out_folder = "D:/0_PHD/Hyrax/BoutGarbageCollector/AUDIO/ACA/GoodTargetGarbage/DATASET/"
    main(Data_folder, Out_folder)