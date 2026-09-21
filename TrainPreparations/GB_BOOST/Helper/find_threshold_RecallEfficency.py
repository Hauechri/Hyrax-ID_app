import os
import glob
import numpy as np

# ------------------------------
# Utility Functions
# ------------------------------

class_map = {0: "chuck", 1: "snort", 2: "wail"}
VALID_LABELS = set(class_map.values())

def load_labels(file_path, is_prediction=False):
    """Load Audacity label files into a list of (start, end, class, score)."""
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end = float(parts[0].replace(",", ".")), float(parts[1].replace(",", "."))
            label = parts[2].lower()
            score = None
            if is_prediction and "_" in label:
                *classname_parts, score_str = label.split("_")
                label = "_".join(classname_parts)
                try:
                    score = float(score_str)
                except ValueError:
                    score = None
            if label not in VALID_LABELS:
                continue
            labels.append((start, end, label, score))
    return labels


def compute_overlap(gt_start, gt_end, pred_start, pred_end):
    """Compute overlap fraction relative to GT event."""
    overlap = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    duration = gt_end - gt_start
    return overlap / duration if duration > 0 else 0


# ------------------------------
# Core Comparison
# ------------------------------
def evaluate_predictions(gt_folder, pred_folder, multiclass=True, overlap_threshold=0.7):
    """
    Evaluate predictions against ground truth on a per-file basis.
    Returns:
        matches: List of (score, overlap) for true positives.
        false_positives: List of scores for false positives.
    """
    gt_files = glob.glob(os.path.join(gt_folder, "*.txt"))

    all_matches = []
    false_positives = []

    for gt_file in gt_files:
        base = os.path.basename(gt_file).replace(".txt", "")
        pred_file = os.path.join(pred_folder, base + "_labels.txt")

        if not os.path.exists(pred_file):
            print(f"[!] Missing prediction file for {base}")
            continue

        # Load labels once per file
        gt_labels = load_labels(gt_file, is_prediction=False)
        pred_labels = load_labels(pred_file, is_prediction=True)

        # Track which predictions have been matched
        pred_used = [False] * len(pred_labels)

        # Match ground truths with predictions
        for gt_start, gt_end, gt_class, _ in gt_labels:
            best_match_idx = -1
            best_overlap = 0
            best_score = None

            for i, (pred_start, pred_end, pred_class, pred_score) in enumerate(pred_labels):
                if pred_used[i]:
                    continue

                if multiclass and gt_class != pred_class:
                    continue

                overlap = compute_overlap(gt_start, gt_end, pred_start, pred_end)

                if overlap >= overlap_threshold and overlap > best_overlap:
                    best_overlap = overlap
                    best_match_idx = i
                    best_score = pred_score

            # Store best match (True Positive)
            if best_match_idx != -1:
                pred_used[best_match_idx] = True
                all_matches.append((best_score, best_overlap))

        # Remaining unmatched predictions are False Positives
        for i, (_, _, _, pred_score) in enumerate(pred_labels):
            if not pred_used[i] and pred_score is not None:
                false_positives.append(pred_score)

    return all_matches, false_positives


def find_best_threshold_recall_efficiency(matches, false_positives, min_recall=0.85):
    """
    Find threshold that maximizes precision gain per recall loss.

    Args:
        min_recall: ignore thresholds below this recall (keeps you in high-recall regime)
    """
    scores_tp = [m[0] for m in matches if m[0] is not None]
    scores_fp = [s for s in false_positives if s is not None]

    if not scores_tp or not scores_fp:
        print("Not enough data to compute threshold.")
        return None

    all_scores = np.linspace(0, 1, 101)
    metrics = []

    # Compute PR points
    for t in all_scores:
        tp = sum(s >= t for s in scores_tp)
        fp = sum(s >= t for s in scores_fp)
        fn = sum(s < t for s in scores_tp)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        metrics.append((t, precision, recall))

    # Sort by recall DESC (high → low)
    metrics.sort(key=lambda x: x[2], reverse=True)

    best_threshold = None
    best_efficiency = -np.inf

    print("\n--- Recall Efficiency Analysis ---")

    for i in range(1, len(metrics)):
        t_prev, p_prev, r_prev = metrics[i - 1]
        t_curr, p_curr, r_curr = metrics[i]

        # Only consider high recall region
        if r_curr < min_recall:
            continue

        delta_p = p_curr - p_prev
        delta_r = r_prev - r_curr  # recall loss

        if delta_r <= 0:
            continue

        efficiency = delta_p / delta_r

        print(f"Thresh={t_curr:.2f} | ΔP={delta_p:.3f} | ΔR={delta_r:.3f} | Eff={efficiency:.3f}")

        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_threshold = t_curr

    print("\n--- Best Threshold (Recall Efficiency) ---")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Efficiency Score: {best_efficiency:.3f}")

    return best_threshold, metrics


# ------------------------------
# Main Script
# ------------------------------
if __name__ == "__main__":
    print("=== Compare Audacity Label Folders ===")
    gt_folder = "D:/0_PHD/Hyrax/ImprovedAnalysis/PoorGoodValidation/ORIGINAL/GT_Good_Bouts/"
    pred_folder = "D:/0_PHD/Hyrax/ImprovedAnalysis/ThesholdGood/ACA/"
    multiclass = True

    matches, false_positives = evaluate_predictions(
        gt_folder, pred_folder, multiclass=multiclass, overlap_threshold=0.5
    )

    find_best_threshold_recall_efficiency(matches, false_positives, min_recall=0.85)
