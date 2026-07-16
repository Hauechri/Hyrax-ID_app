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


def find_best_threshold(matches, false_positives):
    """Find threshold that maximizes F1 score."""
    scores_tp = [m[0] for m in matches if m[0] is not None]
    scores_fp = [s for s in false_positives if s is not None]

    if not scores_tp or not scores_fp:
        print("Not enough data to compute threshold.")
        return None

    all_scores = np.linspace(0, 1, 101)
    best_threshold = 0
    best_f1 = 0
    metrics = []

    for t in all_scores:
        tp = sum(s >= t for s in scores_tp)
        fp = sum(s >= t for s in scores_fp)
        fn = sum(s < t for s in scores_tp)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        metrics.append((t, precision, recall, f1))

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print("\n--- Threshold Optimization Results ---")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"F1 score: {best_f1:.3f}")

    print("\nPrecision / Recall Curve:")
    for t, p, r, f in metrics:  # print every 10th threshold for readability
        print(f"  Thresh={t:.2f} | Prec={p:.2f} | Rec={r:.2f} | F1={f:.2f}")

    return best_threshold, metrics


# ------------------------------
# Main Script
# ------------------------------
if __name__ == "__main__":
    print("=== Compare Audacity Label Folders ===")
    gt_folder = "D:/0_PHD/Hyrax/ImprovedAnalysis/PoorGoodValidation/OLDPREDICTIONS/ORIGINAL/GT_Good_Bouts/"
    pred_folder = "D:/0_PHD/Hyrax/ImprovedAnalysis/ThesholdGood/ACA/"
    multiclass = True

    matches, false_positives = evaluate_predictions(
        gt_folder, pred_folder, multiclass=multiclass, overlap_threshold=0.5
    )

    find_best_threshold(matches, false_positives)
