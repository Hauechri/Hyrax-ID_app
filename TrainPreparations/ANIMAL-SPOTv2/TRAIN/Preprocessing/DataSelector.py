import random
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================
#OUTPUT_AUDIO_FOLDER = 'C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ACASet/Extracts/'
#OUTPUT_LABEL_FOLDER = 'C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ACASet/Labels/'
INPUT_AUDIO_DIR = Path('C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ORGSet/Extracts/')
INPUT_LABEL_DIR = Path('C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ORGSet/Labels/')

OUTPUT_AUDIO_DIR = Path("C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ORGSELECT/SEGMENTS/")
OUTPUT_LABEL_DIR = Path("C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ORGSELECT/LABELS2")

EMPTY_RATIO = 0.10  # 10% of empty samples


# =========================
# HELPERS
# =========================
def is_empty_label(label_path):
    """
    Returns True if label file has no annotations.
    """
    with open(label_path) as f:
        content = f.read().strip()
        return len(content) == 0


# =========================
# MAIN
# =========================
def main():
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(INPUT_LABEL_DIR.glob("*.txt"))

    non_empty = []
    empty = []

    # split dataset
    for label_path in label_files:
        if is_empty_label(label_path):
            empty.append(label_path)
        else:
            non_empty.append(label_path)

    print(f"Total samples: {len(label_files)}")
    print(f"Non-empty: {len(non_empty)}")
    print(f"Empty: {len(empty)}")

    # 🎯 sample empties
    n_select_empty = int(len(non_empty) * EMPTY_RATIO)
    selected_empty = random.sample(empty, n_select_empty)

    print(f"Selected empty samples: {len(selected_empty)}")

    # combine
    selected_labels = non_empty + selected_empty

    # =========================
    # COPY FILES
    # =========================
    for label_path in selected_labels:
        stem = label_path.stem

        # find corresponding audio (.wav or .WAV safe)
        audio_path = None
        for ext in [".wav", ".WAV"]:
            candidate = INPUT_AUDIO_DIR / f"{stem}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if not audio_path:
            print(f"⚠️ Missing audio for {stem}, skipping.")
            continue

        # copy label
        shutil.copy(label_path, OUTPUT_LABEL_DIR / label_path.name)

        # copy audio (normalize to .wav)
        out_audio_path = OUTPUT_AUDIO_DIR / f"{stem}.wav"
        shutil.copy(audio_path, out_audio_path)

    print(f"✅ Done. Final dataset size: {len(selected_labels)}")


if __name__ == "__main__":
    main()