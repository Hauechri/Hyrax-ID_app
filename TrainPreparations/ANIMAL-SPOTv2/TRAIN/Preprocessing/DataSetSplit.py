import os
import shutil
import random
from pathlib import Path

# =========================
# CONFIG (EDIT THESE)
# =========================
#INPUT_FOLDER = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ACASelect/SEGMENTS/"
#OUTPUT_FOLDER = "C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ACASelect/IMAGES/"


IMAGES_DIR = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ORGSELECT/IMAGES/"
LABELS_DIR = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ORGSELECT/LABELS2/"
OUTPUT_DIR = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASET/ORGSet/"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- NEW: MULTICLASS SETUP ---
CLASS_NAMES = ["chuck", "snort", "wail"]

SEED = random.randint(0, 100) %42
# =========================


def prepare_yolo_dataset():
    random.seed(SEED)

    images_dir = Path(IMAGES_DIR)
    labels_dir = Path(LABELS_DIR)
    output_dir = Path(OUTPUT_DIR)

    # Collect images
    image_extensions = [".jpg", ".jpeg", ".png"]
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions]

    # Keep only images that have labels
    images = [img for img in images if (labels_dir / (img.stem + ".txt")).exists()]

    print(f"Found {len(images)} valid image-label pairs.")

    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    # Create folder structure
    for split in splits:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy files # Move instead
    for split, files in splits.items():
        for img_path in files:
            label_path = labels_dir / (img_path.stem + ".txt")

            shutil.move(img_path, output_dir / split / "images" / img_path.name)
            shutil.move(label_path, output_dir / split / "labels" / label_path.name)

    print("Files copied into train/val/test splits.")

    # --- CREATE YAML ---
    names_yaml = "\n".join([f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)])

    yaml_content = f"""path: {output_dir.resolve()}
train: train/images
val: val/images
test: test/images

nc: {len(CLASS_NAMES)}
names:
{names_yaml}
"""

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"data.yaml created at: {yaml_path}")


if __name__ == "__main__":
    prepare_yolo_dataset()