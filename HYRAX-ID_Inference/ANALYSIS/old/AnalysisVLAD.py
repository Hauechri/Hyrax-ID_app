import os
import re
import csv
from collections import defaultdict

# Folder containing Audacity annotation files
FOLDER_PATH = "D:/0_PHD/Hyrax/AnalysisGTBouts/VLAD/"
OUTPUT_FILE = "D:/0_PHD/Hyrax/AnalysisGTBouts/animal_communication_summary.csv"
# Known communication types
CALL_TYPES = ["Wail", "Chuck", "Snort", "Bout"]


import os
import re

def extract_animal_and_year(filename):
    """
    Extract the animal name and year from the filename.
    - Animal name: first part before '_'
    - Year: extracted from any ddmmyy pattern.
    """
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split("_")

    # Animal name is always the first part
    animal = parts[0]

    year = "unknown"
    date_pattern = re.compile(r"^\d{6}$")  # Strict ddmmyy

    for part in parts:
        if date_pattern.match(part):
            day = int(part[:2])
            month = int(part[2:4])
            yy = int(part[4:6])

            # Basic validation
            if 1 <= day <= 31 and 1 <= month <= 12:
                # Convert to four-digit year
                year = 1900 + yy if yy >= 70 else 2000 + yy
                break

    return animal, str(year)


def normalize_label(label):
    """
    Normalize annotation labels to the four communication types.
    """
    label = label.strip()

    if label.startswith("bout"):
        return "Bout"

    for call in ["Wail", "Chuck", "Snort"]:
        if label.lower().startswith(call.lower()):
            return call

    return None  # Ignore unknown labels


def parse_annotation_file(filepath):
    """
    Parse an Audacity annotation file and count communication types.
    Audacity labels typically follow the format:
    start_time <tab> end_time <tab> label
    """
    counts = defaultdict(int)

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            label = parts[2]
            normalized = normalize_label(label)

            if normalized:
                counts[normalized] += 1

    return counts


def analyze_annotations(folder_path):
    """
    Analyze all annotation files in the given folder and aggregate results.
    """
    results = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(folder_path, filename)

        animal, year = extract_animal_and_year(filename)
        counts = parse_annotation_file(filepath)

        for call_type, count in counts.items():
            results[animal][year][call_type] += count

    return results


def save_to_csv(results, output_file="animal_communication_summary.csv"):
    """
    Save the aggregated results to a CSV file.
    """
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(["Animal", "Year", "Wail", "Chuck", "Snort", "Bout"])

        for animal in sorted(results.keys()):
            for year in sorted(results[animal].keys()):
                row = [
                    animal,
                    year,
                    results[animal][year].get("Wail", 0),
                    results[animal][year].get("Chuck", 0),
                    results[animal][year].get("Snort", 0),
                    results[animal][year].get("Bout", 0),
                ]
                writer.writerow(row)


if __name__ == "__main__":
    results = analyze_annotations(FOLDER_PATH)
    save_to_csv(results, output_file=OUTPUT_FILE)

    print("Analysis complete!")
    print("Results saved to 'animal_communication_summary.csv'.")