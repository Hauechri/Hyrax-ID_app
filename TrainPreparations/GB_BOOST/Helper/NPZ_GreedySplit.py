import os
import shutil
import random
from collections import defaultdict

import numpy as np

# ==========================================================
# SETTINGS
# ==========================================================

INPUT_FOLDER = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/BoostTrain/NPZ/"
OUTPUT_FOLDER = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/BoostTrain/BoostDataset/"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)

# ==========================================================
# LOAD DATA
# ==========================================================

animal_files = defaultdict(list)

for file in os.listdir(INPUT_FOLDER):

    if not file.endswith(".npz"):
        continue

    animal = file.split("_")[0]

    path = os.path.join(INPUT_FOLDER, file)

    data = np.load(path, allow_pickle=True)

    animal_files[animal].append({
        "filename": file,
        "path": path,
        "samples": len(data["y"])
    })

# ==========================================================
# OUTPUT FOLDERS
# ==========================================================

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_FOLDER, split), exist_ok=True)


# ==========================================================
# OPTIMIZER
# ==========================================================

def objective(split_dict, targets):
    """Smaller is better."""

    score = 0

    for split in ["train", "val", "test"]:

        samples = sum(r["samples"] for r in split_dict[split])

        score += (samples - targets[split]) ** 2

    return score


summary = {
    "train": defaultdict(int),
    "val": defaultdict(int),
    "test": defaultdict(int)
}

# ==========================================================
# PROCESS EACH ANIMAL
# ==========================================================

for animal, recordings in animal_files.items():

    random.shuffle(recordings)

    total = sum(r["samples"] for r in recordings)

    targets = {
        "train": total * TRAIN_RATIO,
        "val": total * VAL_RATIO,
        "test": total * TEST_RATIO
    }

    # ------------------------------------------------------
    # Initial split (by recordings only)
    # ------------------------------------------------------

    n = len(recordings)

    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)

    split = {
        "train": recordings[:n_train],
        "val": recordings[n_train:n_train+n_val],
        "test": recordings[n_train+n_val:]
    }

    # ------------------------------------------------------
    # Hill climbing
    # ------------------------------------------------------

    improved = True

    while improved:

        improved = False

        current_score = objective(split, targets)

        best_score = current_score
        best_move = None

        for src in ["train", "val", "test"]:

            for idx, rec in enumerate(split[src]):

                for dst in ["train", "val", "test"]:

                    if src == dst:
                        continue

                    candidate = {
                        "train": split["train"][:],
                        "val": split["val"][:],
                        "test": split["test"][:]
                    }

                    candidate[src].pop(idx)
                    candidate[dst].append(rec)

                    score = objective(candidate, targets)

                    if score < best_score:

                        best_score = score
                        best_move = (src, dst, idx)

        if best_move is not None:

            src, dst, idx = best_move

            rec = split[src].pop(idx)
            split[dst].append(rec)

            improved = True

    # ------------------------------------------------------
    # Copy files
    # ------------------------------------------------------

    print(f"\n{animal}")

    for split_name in ["train", "val", "test"]:

        samples = sum(r["samples"] for r in split[split_name])

        print(f"{split_name:5s}: {samples}")

        summary[split_name][animal] = samples

        for rec in split[split_name]:

            shutil.copy2(
                rec["path"],
                os.path.join(
                    OUTPUT_FOLDER,
                    split_name,
                    rec["filename"]
                )
            )

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n==============================")

overall = {}

grand_total = 0

for split in ["train", "val", "test"]:

    overall[split] = sum(summary[split].values())
    grand_total += overall[split]

for split in ["train", "val", "test"]:

    print(
        f"{split:5s}: "
        f"{overall[split]:8d} samples "
        f"({100*overall[split]/grand_total:5.2f}%)"
    )