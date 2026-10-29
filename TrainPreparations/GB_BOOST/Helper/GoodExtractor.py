import os
import glob
import numpy as np
from collections import defaultdict
import soundfile as sf

class Element:
    def __init__(self, start, end, label, bout_id, conf=1.0):
        self.start = start
        self.end = end
        self.label = label
        self.bout_id = bout_id
        self.conf = conf


# Global counter for unique bout IDs across files
global_bout_counter = 0


def extract_snippet(audio, sr, start, end, pad=0.2):
    start = max(0, start - pad)
    end = end + pad
    return audio[int(start * sr):int(end * sr)]


def compute_iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0


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


def main(gt_folder, pred_folder, audio_folder, out_target, out_fp):
    os.makedirs(out_target, exist_ok=True)
    os.makedirs(out_fp, exist_ok=True)

    for gt_file in glob.glob(os.path.join(gt_folder, "*.txt")):
        base = os.path.basename(gt_file).replace(".txt", "")
        pred_file = os.path.join(pred_folder, base + "_labels.txt")
        audio_file = os.path.join(audio_folder, base + ".wav")

        if not os.path.exists(pred_file) or not os.path.exists(audio_file):
            continue

        # Parse
        gt_elements, _ = parse_file(gt_file, False)
        pred_elements, _ = parse_file(pred_file, True)

        # Match per file
        TP, FP, FN, MS, matches = match_elements(gt_elements, pred_elements)

        matched_pred = set(j for _, j, _ in matches)

        # Load audio
        audio, sr = sf.read(audio_file)

        # ------------------
        # SAVE TARGETS (TP + MS)
        # ------------------
        for i, j, iou in matches:
            pred_elem = pred_elements[j]
            gt_elem = gt_elements[i]

            snippet = extract_snippet(audio, sr, pred_elem.start, pred_elem.end)

            # Distinguish TP vs MS
            if gt_elem.label == pred_elem.label:
                tag = "TP"
            else:
                tag = "MS"

            out_path = os.path.join(
                out_target,
                f"{base}___{tag}_{j}_start{pred_elem.start:.3f}_end{pred_elem.end:.3f}_{pred_elem.label}_conf{pred_elem.conf:.2f}_iou{iou:.2f}.wav"
            )
            sf.write(out_path, snippet, sr)

        # ------------------
        # SAVE GARBAGE (FP)
        # ------------------
        for j, elem in enumerate(pred_elements):
            if j in matched_pred:
                continue

            snippet = extract_snippet(audio, sr, elem.start, elem.end)

            out_path = os.path.join(
                out_fp,
                f"{base}___FP_{j}_start{elem.start:.3f}_end{elem.end:.3f}_{elem.label}_conf{elem.conf:.2f}.wav"
            )
            sf.write(out_path, snippet, sr)

    print("Done extracting TP / MS / FP dataset.")


if __name__ == "__main__":
    GT_FOLDER = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/GTLabels/"
    PRED_FOLDER = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/Predicts/ACA/"
    AUDIO_FOLDER = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/ACA/"
    OUTPUT_TARGET = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/BoostTrain/Target/"
    OUTPUT_FP = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/BoostTrain/FP/"
    main(GT_FOLDER, PRED_FOLDER, AUDIO_FOLDER, OUTPUT_TARGET, OUTPUT_FP)



