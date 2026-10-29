import os
import glob
import numpy as np
from collections import defaultdict

# =========================
# CONFIG
# =========================
IOU_THRESHOLD = 0.5
CLASS_NAMES = ["chuck", "snort", "wail"]

# =========================
# DATA STRUCTURES
# =========================
class Element:
    def __init__(self, start, end, label, bout_id, conf=1.0):
        self.start = start
        self.end = end
        self.label = label
        self.bout_id = bout_id
        self.conf = conf


# =========================
# PARSING
# =========================
# Global counter for unique bout IDs across files
global_bout_counter = 0

def parse_file(filepath, is_pred=False):
    """
    Parse a file into elements and bouts.
    Assign elements to bouts using a global counter for unique bout IDs.
    """
    global global_bout_counter
    elements = []
    bouts = []

    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            start = float(parts[0].replace(",", "."))
            end = float(parts[1].replace(",", "."))
            label = parts[2].lower()

            # Bout lines
            if "bout" in label:
                bouts.append((start, end))
                continue

            # Element lines
            conf = 1.0
            if is_pred and "_" in label:
                *label_parts, conf_str = label.split("_")
                label = "_".join(label_parts)
                try:
                    conf = float(conf_str)
                except:
                    conf = 0.0

            elements.append(Element(start, end, label, bout_id=-1, conf=conf))

    # --- Sort elements and bouts by start time ---
    elements.sort(key=lambda x: x.start)
    bouts.sort(key=lambda x: x[0])

    # --- Assign bout IDs to elements ---
    for bout_start, bout_end in bouts:
        global_bout_counter += 1  # one ID per bout
        for elem in elements:
            if bout_start <= elem.start <= bout_end:
                elem.bout_id = global_bout_counter

    for elem in elements:
        if elem.bout_id == -1:
            print("stop")

    return elements, bouts


# =========================
# IOU
# =========================
def compute_iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0


# =========================
# MATCHING
# =========================
def match_elements(gt_elements, pred_elements, iou_thr=0.5):
    matched_gt = set()
    matched_pred = set()
    matches = []

    pairs = []
    for i, gt in enumerate(gt_elements):
        for j, pred in enumerate(pred_elements):
            iou = compute_iou(gt.start, gt.end, pred.start, pred.end)
            if iou >= iou_thr:
                pairs.append((iou, i, j))

    pairs.sort(reverse=True, key=lambda x: x[0])

    for iou, i, j in pairs:
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        matches.append((i, j, iou))

    TP, FP, FN, MS = 0, 0, 0, 0

    for i, j, _ in matches:
        if gt_elements[i].label == pred_elements[j].label:
            TP += 1
        else:
            MS += 1

    FP = len(pred_elements) - len(matches)
    FN = len(gt_elements) - len(matches)

    return TP, FP, FN, MS, matches

def compute_per_class_metrics(gt_elements, pred_elements, matches):
    per_class = {
        c: {"TP": 0, "FP": 0, "FN": 0, "MS": 0}
        for c in CLASS_NAMES
    }

    matched_gt = {i for i, _, _ in matches}
    matched_pred = {j for _, j, _ in matches}

    # --- matched pairs ---
    for i, j, _ in matches:
        gt_c = gt_elements[i].label
        pred_c = pred_elements[j].label

        if gt_c == pred_c:
            per_class[gt_c]["TP"] += 1
        else:
            per_class[gt_c]["MS"] += 1
            per_class[pred_c]["MS"] += 1

    # --- FN ---
    for i, gt in enumerate(gt_elements):
        if i not in matched_gt:
            per_class[gt.label]["FN"] += 1

    # --- FP ---
    for j, pred in enumerate(pred_elements):
        if j not in matched_pred:
            per_class[pred.label]["FP"] += 1

    return per_class

# =========================
# METRICS
# =========================
def compute_metrics(TP, FP, FN, MS):
    precision = TP / (TP + FP + MS + 1e-9)
    recall = TP / (TP + FN + MS + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


# =========================
# AP
# =========================
def compute_ap_curve(precision, recall):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def compute_ap(gt, pred, iou_thr):
    if not pred:
        return 0.0

    pred = sorted(pred, key=lambda x: -x.conf)

    tp, fp = [], []
    matched = set()

    for p in pred:
        best_iou = 0
        best_j = -1

        for j, g in enumerate(gt):
            if j in matched:
                continue
            iou = compute_iou(p.start, p.end, g.start, g.end)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thr and p.label == gt[best_j].label:
            tp.append(1)
            fp.append(0)
            matched.add(best_j)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / (len(gt) + 1e-9)
    precision = tp / (tp + fp + 1e-9)

    return compute_ap_curve(precision, recall)

def compute_per_class_ap(gt_elements, pred_elements):
    ap_results = {}

    for c in CLASS_NAMES:
        gt_c = [g for g in gt_elements if g.label == c]
        pred_c = [p for p in pred_elements if p.label == c]

        ap50 = compute_ap(gt_c, pred_c, 0.5)
        aps = [compute_ap(gt_c, pred_c, t) for t in np.arange(0.5, 1.0, 0.05)]
        ap5095 = np.mean(aps)

        ap_results[c] = {
            "AP@0.5": ap50,
            "AP@0.5:0.95": ap5095
        }

    return ap_results

# =========================
# GRAPH
# =========================
def build_graph(matches, gt_elements, pred_elements):
    """
    Build a graph connecting GT bouts and Pred bouts based on element matches.
    All bouts are included, even if unmatched.
    """
    graph = defaultdict(set)

    # Add all GT bouts as nodes
    gt_bout_ids = set(e.bout_id for e in gt_elements)
    for bid in gt_bout_ids:
        graph[("gt", bid)]  # ensures node exists

    # Add all Pred bouts as nodes
    pred_bout_ids = set(e.bout_id for e in pred_elements)
    for bid in pred_bout_ids:
        graph[("pred", bid)]  # ensures node exists

    # Connect bouts that share matched elements
    for i, j, _ in matches:
        g_node = ("gt", gt_elements[i].bout_id)
        p_node = ("pred", pred_elements[j].bout_id)
        graph[g_node].add(p_node)
        graph[p_node].add(g_node)

    return graph


def get_components(graph):
    visited = set()
    components = []

    for node in graph:
        if node in visited:
            continue

        stack = [node]
        comp = set()

        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            stack.extend(graph[n])

        components.append(comp)

    return components


# =========================
# LEVENSHTEIN
# =========================
def levenshtein_ops(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    i, j = m, n
    ops = {"insert": 0, "delete": 0, "substitute": 0}

    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops["delete"] += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops["insert"] += 1
            j -= 1
        else:
            if i > 0 and j > 0:
                if seq1[i-1] != seq2[j-1]:
                    ops["substitute"] += 1
                i -= 1
                j -= 1

    return dp[m][n], ops


# =========================
# BOUT EVAL
# =========================
def evaluate_bouts(components, gt_elements, pred_elements):
    """
    Evaluate bouts using Levenshtein distance between sequences of element labels
    within each connected component of bouts, with weighted averaging.
    """
    total_ops = {"insert": 0, "delete": 0, "substitute": 0}
    weighted_distances = []

    total_gt_len = 0
    total_pred_len = 0
    total_align_len = 0

    for comp in components:
        # --- GT and Pred bout IDs in this component ---
        gt_ids = {bid for t, bid in comp if t == "gt"}
        pred_ids = {bid for t, bid in comp if t == "pred"}

        # --- Flatten elements, sorted by start time ---
        gt_seq = sorted([e for e in gt_elements if e.bout_id in gt_ids], key=lambda x: x.start)
        pred_seq = sorted([e for e in pred_elements if e.bout_id in pred_ids], key=lambda x: x.start)

        gt_labels = [e.label for e in gt_seq]
        pred_labels = [e.label for e in pred_seq]

        len_gt = len(gt_labels)
        len_pred = len(pred_labels)
        align_len = max(len_gt, len_pred, 1)

        total_gt_len += len_gt
        total_pred_len += len_pred
        total_align_len += align_len

        # --- Compute Levenshtein distance and ops ---
        dist, ops = levenshtein_ops(gt_labels, pred_labels)

        # --- Weighted normalized distance ---
        norm = dist / align_len
        weighted_distances.append((norm, align_len))  # weight by component length

        # --- Accumulate ops weighted by component size ---
        for k in total_ops:
            total_ops[k] += ops[k]

    # --- Weighted average Levenshtein ---
    if weighted_distances:
        avg_dist = sum(d * w for d, w in weighted_distances) / sum(w for _, w in weighted_distances)
    else:
        avg_dist = 0

    # --- Normalized rates ---
    insertion_rate = total_ops["insert"] / (total_pred_len + 1e-9)
    deletion_rate  = total_ops["delete"] / (total_gt_len + 1e-9)
    substitution_rate = total_ops["substitute"] / (total_gt_len + 1e-9)

    return {
        "avg_lev": avg_dist,
        "ops": total_ops,
        "total_gt": total_gt_len,
        "total_pred": total_pred_len,
        "total_align": total_align_len,
        "insertion_rate": insertion_rate,
        "deletion_rate": deletion_rate,
        "substitution_rate": substitution_rate
    }


# =========================
# MAIN
# =========================
def main(gt_folder, pred_folder):
    all_gt, all_pred = [], []

    for gt_file in glob.glob(os.path.join(gt_folder, "*.txt")):
        base = os.path.basename(gt_file).replace(".txt", "")
        pred_file = os.path.join(pred_folder, base + "_labels.txt")

        if not os.path.exists(pred_file):
            continue

        gt, _ = parse_file(gt_file, False)
        pred, _ = parse_file(pred_file, True)

        all_gt.extend(gt)
        all_pred.extend(pred)

    # =========================
    # PER-CLASS METRICS
    # =========================
    TP, FP, FN, MS, matches = match_elements(all_gt, all_pred)

    per_class = compute_per_class_metrics(all_gt, all_pred, matches)
    per_class_ap = compute_per_class_ap(all_gt, all_pred)

    print("\n=== PER-CLASS ELEMENT RESULTS ===")

    for c in CLASS_NAMES:
        TPc = per_class[c]["TP"]
        FPc = per_class[c]["FP"]
        FNc = per_class[c]["FN"]
        MSc = per_class[c]["MS"]

        p_c, r_c, f1_c = compute_metrics(TPc, FPc, FNc, MSc)

        print(f"\nClass: {c}")
        print(f"TP: {TPc}  FP: {FPc}  FN: {FNc}  MS: {MSc}")
        print(f"Precision: {p_c:.4f}")
        print(f"Recall:    {r_c:.4f}")
        print(f"F1:        {f1_c:.4f}")
        print(f"AP@0.5:     {per_class_ap[c]['AP@0.5']:.4f}")
        print(f"AP@0.5:0.95:{per_class_ap[c]['AP@0.5:0.95']:.4f}")



    print("\n=== ELEMENT RESULTS ===")
    print("TP:", TP, "FP:", FP, "FN:", FN, "MS:", MS)

    p, r, f1 = compute_metrics(TP, FP, FN, MS)
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")

    # AP
    ap50 = compute_ap(all_gt, all_pred, 0.5)
    aps = [compute_ap(all_gt, all_pred, t) for t in np.arange(0.5, 1.0, 0.05)]
    ap5095 = np.mean(aps)

    print(f"mAP@0.5:     {ap50:.4f}")
    print(f"mAP@0.5:0.95:{ap5095:.4f}")

    # GRAPH
    graph = build_graph(matches, all_gt, all_pred)
    comps = get_components(graph)

    debug_count = sum(1 for elem in all_gt if elem.bout_id == -1)
    debug_count2 = sum(1 for elem in all_pred if elem.bout_id == -1)
    bout_res = evaluate_bouts(comps, all_gt, all_pred)

    print("\n=== BOUT RESULTS ===")
    print("Total GT elements:   ", bout_res["total_gt"])
    print("Total Pred elements: ", bout_res["total_pred"])
    print("Total Align length:  ", bout_res["total_align"])

    print("\nInsertions (FP):     ", bout_res["ops"]["insert"])
    print("Deletions (FN):      ", bout_res["ops"]["delete"])
    print("Substitutions (MS):  ", bout_res["ops"]["substitute"])

    # --- normalized error rates (directly from evaluate_bouts) ---
    print("\n--- Normalized ---")
    print("Insertion rate:     ", round(bout_res["insertion_rate"], 4))
    print("Deletion rate:      ", round(bout_res["deletion_rate"], 4))
    print("Substitution rate:  ", round(bout_res["substitution_rate"], 4))

    print("\nAvg Levenshtein:    ", round(bout_res["avg_lev"], 3))


if __name__ == "__main__":
    GT_FOLDER = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/ValidateData/GTLabels/"
    PRED_FOLDER = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/ValidateData/Predict_GB/ACA_03_03/"

    main(GT_FOLDER, PRED_FOLDER)