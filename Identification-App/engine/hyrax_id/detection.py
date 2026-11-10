"""Detector -> merge -> GarbageCollector filter -> bouts, for one in-memory signal."""
import numpy as np

from .garbage_filter import compute_features, features_to_matrix
from .postprocess import merge_overlapping_labels, convert_to_detections, group_bouts


def detect_elements_for_file(detector, garbagecollector, trained_feature_keys, signal, sr, hop_samples,
                             interarrival_threshold, context_windowsize, GB_threshold, GB_scoreweight):
    """Returns (elements [(start, end, cls_name, conf)], bouts [(start, end, "BOUT", n)])."""
    raw_labels = detector.detect(signal, sr, hop_samples)
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
