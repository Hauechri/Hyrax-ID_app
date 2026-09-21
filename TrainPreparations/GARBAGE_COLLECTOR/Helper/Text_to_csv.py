import os
import re
import csv
from pathlib import Path

# --------------------------------------------------
# Settings
# --------------------------------------------------
INPUT_FOLDER = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/ValidateData/FitGBScoreThresh_Boost/RESULTS/"
OUTPUT_CSV = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/ValidateData/FitGBScoreThresh_Boost/res.csv"

# --------------------------------------------------
# Regular expressions
# --------------------------------------------------
class_pattern = re.compile(
    r"Class:\s*(\w+)\s*"
    r"TP:\s*(\d+),\s*FP:\s*(\d+),\s*FN:\s*(\d+)\s*"
    r"Precision:\s*([\d.]+)\s*"
    r"Recall:\s*([\d.]+)\s*"
    r"F1:\s*([\d.]+)\s*"
    r"AP@0\.5:\s*([\d.]+)\s*"
    r"AP@0\.5:0\.95:\s*([\d.]+)",
    re.MULTILINE,
)

overall_pattern = re.compile(
    r"--- Overall metrics ---\s*"
    r"Precision:\s*([\d.]+)\s*"
    r"Recall:\s*([\d.]+)\s*"
    r"F1:\s*([\d.]+)\s*"
    r"mAP@0\.5:\s*([\d.]+)\s*"
    r"mAP@0\.5:0\.95:\s*([\d.]+)",
    re.MULTILINE,
)

rows = []

for txt_file in sorted(Path(INPUT_FOLDER).glob("*.txt")):

    with open(txt_file, "r") as f:
        text = f.read()

    row = {"File": txt_file.stem}

    # -------------------------
    # Per-class metrics
    # -------------------------
    for match in class_pattern.finditer(text):
        cls = match.group(1)

        row[f"{cls}_TP"] = int(match.group(2))
        row[f"{cls}_FP"] = int(match.group(3))
        row[f"{cls}_FN"] = int(match.group(4))
        row[f"{cls}_Precision"] = float(match.group(5))
        row[f"{cls}_Recall"] = float(match.group(6))
        row[f"{cls}_F1"] = float(match.group(7))
        row[f"{cls}_AP50"] = float(match.group(8))
        row[f"{cls}_AP5095"] = float(match.group(9))

    # -------------------------
    # Overall metrics
    # -------------------------
    overall = overall_pattern.search(text)

    if overall:
        row["Overall_Precision"] = float(overall.group(1))
        row["Overall_Recall"] = float(overall.group(2))
        row["Overall_F1"] = float(overall.group(3))
        row["Overall_mAP50"] = float(overall.group(4))
        row["Overall_mAP5095"] = float(overall.group(5))

    rows.append(row)

# --------------------------------------------------
# Write CSV
# --------------------------------------------------
# Collect all column names
fieldnames = sorted({k for row in rows for k in row.keys()})

# Put File first
fieldnames.remove("File")
fieldnames = ["File"] + fieldnames

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to")
print(OUTPUT_CSV)