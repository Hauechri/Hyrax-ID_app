import os
import pandas as pd
from collections import defaultdict

# ==============================
# CONFIGURATION
# ==============================
ANNOTATION_FOLDER = "D:/0_PHD/Hyrax/AnalysisGTBouts/Heloise/"
EXCEL_FILE = "D:/0_PHD/Hyrax/AnalysisGTBouts/XLSX/TableOfSongAnalysis.xlsx"
OUTPUT_FILE = "D:/0_PHD/Hyrax/AnalysisGTBouts/animal_communication_Heloise.csv"

SONG_COLUMN = "Song"
YEAR_COLUMN = "Year"

CALL_TYPES = ["Wail", "Chuck", "Snort", "Bout"]


# ==============================
# LOAD METADATA FROM EXCEL
# ==============================
def load_metadata(excel_path):
    """
    Load the Excel file and create a lookup dictionary:
    {song_identifier: year}
    """
    df = pd.read_excel(excel_path, engine="openpyxl")

    # Ensure required columns exist
    if SONG_COLUMN not in df.columns or YEAR_COLUMN not in df.columns:
        raise ValueError(
            f"Excel file must contain '{SONG_COLUMN}' and '{YEAR_COLUMN}' columns."
        )

    # Clean data
    df = df[[SONG_COLUMN, YEAR_COLUMN]].dropna()
    df[SONG_COLUMN] = df[SONG_COLUMN].astype(str).str.strip()
    df[YEAR_COLUMN] = df[YEAR_COLUMN].astype(str).str.strip()

    # Create lookup dictionary
    lookup = dict(zip(df[SONG_COLUMN], df[YEAR_COLUMN]))
    return lookup


# ==============================
# NORMALIZE LABELS
# ==============================
def normalize_label(label):
    """
    Normalize annotation labels to the four communication types.
    """
    label = label.strip()

    if label.lower().startswith("bout"):
        return "Bout"

    for call in ["Wail", "Chuck", "Snort"]:
        if label.lower().startswith(call.lower()):
            return call

    return None


# ==============================
# PARSE AUDACITY ANNOTATION FILE
# ==============================
def parse_annotation_file(filepath):
    """
    Parse an Audacity annotation file and count communication types.
    Expected format: start_time <tab> end_time <tab> label
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


# ==============================
# EXTRACT SONG IDENTIFIER
# ==============================
def extract_song_identifier(filename):
    """
    Extract the song identifier from the filename.
    Assumes the identifier is the filename without the extension.
    """
    return os.path.splitext(filename)[0]


# ==============================
# MAIN ANALYSIS FUNCTION
# ==============================
def analyze_annotations(annotation_folder, metadata_lookup):
    """
    Analyze annotation files using metadata from Excel.
    """
    results = defaultdict(lambda: defaultdict(int))

    for filename in os.listdir(annotation_folder):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(annotation_folder, filename)
        song_id = extract_song_identifier(filename)

        # Lookup year from Excel
        year = metadata_lookup.get(song_id, "unknown")

        # Parse annotation file
        counts = parse_annotation_file(filepath)

        # Aggregate results
        key = (song_id, year)
        for call_type, count in counts.items():
            results[key][call_type] += count

    return results


# ==============================
# SAVE RESULTS TO CSV
# ==============================
def save_to_csv(results, output_file):
    """
    Save aggregated results to a CSV file.
    """
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        header = ["Song", "Year", "Wail", "Chuck", "Snort", "Bout"]
        f.write(",".join(header) + "\n")

        for (song, year) in sorted(results.keys()):
            row = [
                song,
                year,
                str(results[(song, year)].get("Wail", 0)),
                str(results[(song, year)].get("Chuck", 0)),
                str(results[(song, year)].get("Snort", 0)),
                str(results[(song, year)].get("Bout", 0)),
            ]
            f.write(",".join(row) + "\n")


# ==============================
# MAIN EXECUTION
# ==============================
def main():
    metadata_lookup = load_metadata(EXCEL_FILE)
    results = analyze_annotations(ANNOTATION_FOLDER, metadata_lookup)
    save_to_csv(results, OUTPUT_FILE)

    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()