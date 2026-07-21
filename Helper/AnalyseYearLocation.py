from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

input_dir = Path("C:/PHD/SideProjectsData/IcasspExperiments/YearLocation/2-4_Arugot/GTLabels/")
output_file_summary = Path("C:/PHD/SideProjectsData/IcasspExperiments/YearLocation/2-4_Arugot/Analysis/2-4_Arugot_file_summary.csv")
output_file_summary.parent.mkdir(parents=True, exist_ok=True)
output_identifier_summary = Path("C:/PHD/SideProjectsData/IcasspExperiments/YearLocation/2-4_Arugot/Analysis/2-4_Arugot_identifier_summary.csv")
output_identifier_summary.parent.mkdir(parents=True, exist_ok=True)

ELEMENT_CLASSES = {"wail", "chuck", "snort"}
BOUT_LABEL = "bout"

# ---------------------------------------------------------------------

file_results = []

for txt_file in input_dir.glob("*.txt"):

    identifier = txt_file.stem.split("_")[0]

    counts = {
        "wail": 0,
        "chuck": 0,
        "snort": 0,
        "bout": 0
    }

    with open(txt_file, "r") as f:
        for line in f:

            parts = line.strip().split("\t")

            if len(parts) < 3:
                continue

            label = parts[2].strip().lower()
            label = label.split("_")[0]

            if label in ELEMENT_CLASSES:
                counts[label] += 1

            elif label == BOUT_LABEL:
                counts["bout"] += 1

    file_results.append({
        "Identifier": identifier,
        "File": txt_file.name,
        "Bouts": counts["bout"],
        "Wails": counts["wail"],
        "Chucks": counts["chuck"],
        "Snorts": counts["snort"],
        "Elements": counts["wail"] + counts["chuck"] + counts["snort"]
    })

# ---------------------------------------------------------------------
# Per-file dataframe
# ---------------------------------------------------------------------

df_files = pd.DataFrame(file_results)

# ---------------------------------------------------------------------
# Per-Identifier summary
# ---------------------------------------------------------------------

df_identifier = (
    df_files
    .groupby("Identifier")
    .agg(
        Files=("File", "count"),
        Bouts=("Bouts", "sum"),
        Wails=("Wails", "sum"),
        Chucks=("Chucks", "sum"),
        Snorts=("Snorts", "sum"),
        Elements=("Elements", "sum")
    )
    .reset_index()
)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

df_files.to_csv(output_file_summary, index=False)
df_identifier.to_csv(output_identifier_summary, index=False)

print(df_files)
print()
print(df_identifier)

print("\nDone.")