import numpy as np


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


def features_to_matrix(feature_list):
    X = []
    for f in feature_list:
        X.append([f[k] for k in FEATURE_KEYS])
    return np.array(X, dtype=np.float32)


def build_dataset(feature_list, labels):
    """
    labels: list of 0 (FP) or 1 (valid)
    """
    X = features_to_matrix(feature_list)
    y = np.array(labels, dtype=np.int32)
    return X, y