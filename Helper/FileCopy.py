from pathlib import Path
import shutil


#2002-2004-Arugot: "O1", "Q7", "R3"
#2007-2013-Arugot: "J9", "Kashtan", "P0", "P1", "T9", "U7"
#2002-2004-David: "M9", "O7", "T1"
#2007-2013-David: "U9", "M0", "P8", "T0", "W4", "X0"

if __name__ == "__main__":
    SelectedGroups = ["O1", "Q7", "R3"]

    input_dir = Path("C:/PHD/SideProjects/Vlad214/DATABASE/GROUND-TRUTH_AUDACITY/")
    output_dir = Path("C:/PHD/SideProjectsData/IcasspExperiments/YearLocation/2-4_Arugot/GTLabels/")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    for wav_file in input_dir.glob("*.txt"):
        if any(wav_file.name.startswith(f"{group}_") for group in SelectedGroups):
            shutil.copy2(wav_file, output_dir)
            copied += 1
            print(f"Copied: {wav_file.name}")

    print(f"\nDone! Copied {copied} files.")