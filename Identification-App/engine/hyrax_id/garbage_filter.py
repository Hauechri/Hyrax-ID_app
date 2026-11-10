"""XGBoost 'GarbageCollector' feature building (extracted unchanged).
NOTE: LABEL_MAP is keyed by class NAME, so the detector's class index order does not matter here."""
from collections import Counter

import numpy as np

LABEL_MAP = {"chuck": 0, "snort": 1, "wail": 2}


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
