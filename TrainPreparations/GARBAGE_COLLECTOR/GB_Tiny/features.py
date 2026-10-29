import numpy as np


def safe_get(arr, idx, default=0.0):
    if idx < 0 or idx >= len(arr):
        return default
    return arr[idx]


def compute_features(detections):
    """
    detections: list of dicts with keys:
        start, end, conf, class_id
    """

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
            # current
            "conf": confs[i],
            "duration": durations[i],
            "class_id": classes[i],

            # temporal
            "dt_prev": dt_prev,
            "dt_next": dt_next,

            # neighbors (confidence)
            "conf_prev1": safe_get(confs, i - 1),
            "conf_prev2": safe_get(confs, i - 2),
            "conf_next1": safe_get(confs, i + 1),
            "conf_next2": safe_get(confs, i + 2),

            # wider gaps
            "dt_prev2": starts[i] - ends[i - 2] if i > 1 else 999.0,
            "dt_next2": starts[i + 2] - ends[i] if i < N - 2 else 999.0,
        }

        features.append(feat)

    return features