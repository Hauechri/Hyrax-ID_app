import os
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

# --- CONFIGURATION ---
AUDIO_FOLDER = 'C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/BIODA/'
LABEL_FOLDER = 'C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/GTLabels/'

OUTPUT_AUDIO_FOLDER = 'C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ORGSet/Extracts/'
OUTPUT_LABEL_FOLDER = 'C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/DATASELECTTMP/ORGSet/Labels/'

SAMPLE_RATE = 48000
SNIPPET_DURATION = 1.0
SNIPPET_SAMPLES = int(SAMPLE_RATE * SNIPPET_DURATION)

HOP_SAMPLES = SNIPPET_SAMPLES
MIN_OVERLAP_RATIO = 0.25

# --- CLASS CONFIG ---
CLASS_MAP = {
    "chuck": 0,
    "snort": 1,
    "wail": 2
}

MODE = "multiclass"   # "multiclass" or "binary"

# --- SETUP ---
os.makedirs(OUTPUT_AUDIO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)

audio_files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith('.wav')])
counter = 1


def read_audacity_labels(label_path):
    labels = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 3:
                continue

            try:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2].lower()
            except:
                continue

            labels.append((label, start, end))

    return labels


# --- MAIN LOOP ---
for audio_file in tqdm(audio_files, desc="Processing audio files"):
    base_name = Path(audio_file).stem
    audio_path = os.path.join(AUDIO_FOLDER, audio_file)
    label_path = os.path.join(LABEL_FOLDER, base_name + '.txt')

    if not os.path.isfile(label_path):
        print(f"[!] No label file for {audio_file}, skipping.")
        continue

    # --- LOAD AUDIO ---
    try:
        audio_data, sr = sf.read(audio_path)
    except Exception as e:
        print(f"[!] Error reading {audio_file}: {e}")
        continue

    if sr != SAMPLE_RATE:
        print(f"[!] Wrong sample rate in {audio_file}: {sr}")
        continue

    if audio_data.ndim != 1:
        print(f"[!] Not mono: {audio_file}")
        continue

    total_samples = len(audio_data)

    # --- LOAD LABELS ---
    labels = read_audacity_labels(label_path)

    # --- SEGMENTATION ---
    for start_sample in range(0, total_samples, HOP_SAMPLES):
        end_sample = start_sample + SNIPPET_SAMPLES

        if end_sample > total_samples:
            break

        snippet = audio_data[start_sample:end_sample]

        snippet_start_sec = start_sample / SAMPLE_RATE
        snippet_end_sec = end_sample / SAMPLE_RATE

        snippet_labels = []

        for label, label_start, label_end in labels:

            if label not in CLASS_MAP:
                continue  # skip unknown labels

            overlap_start = max(label_start, snippet_start_sec)
            overlap_end = min(label_end, snippet_end_sec)

            overlap_duration = max(0.0, overlap_end - overlap_start)
            label_duration = max(0.0, label_end - label_start)

            if label_duration == 0:
                continue

            if overlap_duration / label_duration < MIN_OVERLAP_RATIO:
                continue

            # --- Convert class ---
            if MODE == "binary":
                class_id = 0
            else:
                class_id = CLASS_MAP[label]

            # --- Convert to YOLO ---
            clipped_start = max(label_start, snippet_start_sec)
            clipped_end = min(label_end, snippet_end_sec)

            center = ((clipped_start + clipped_end) / 2) - snippet_start_sec
            width = clipped_end - clipped_start

            x_center = center / SNIPPET_DURATION
            width_norm = width / SNIPPET_DURATION

            snippet_labels.append(
                f"{class_id} {x_center:.6f} 0.5 {width_norm:.6f} 1.0"
            )

        # --- SAVE ---
        out_wav = os.path.join(OUTPUT_AUDIO_FOLDER, f"{counter}.wav")
        out_txt = os.path.join(OUTPUT_LABEL_FOLDER, f"{counter}.txt")

        sf.write(out_wav, snippet, SAMPLE_RATE)

        with open(out_txt, 'w') as f:
            f.write("\n".join(snippet_labels))

        counter += 1