import os
import numpy as np
from pathlib import Path
import soundfile as sf
import torch
import torchaudio

SR = 48000


SPEAKER_LIST = [ 'J9', 'Kashtan', 'M0', 'M9', 'O1', 'O7', 'P0', 'P1', 'P8', 'Q7', 'R3', 'T0', 'T1', 'T9', 'U7', 'U9', 'W4', 'X0' ]
SPEAKER_TO_ID = {
    speaker: idx for idx, speaker in enumerate(sorted(SPEAKER_LIST))
}
# -------------------------
# Parsing
# -------------------------
def parse_annotation_file(path):
    elements = []
    bouts = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            start, end, label = line.split("\t")

            start = float(start)
            end = float(end)

            if label.startswith("BOUT"):
                bout_id = int(float(label.split("_")[1]))

                bouts.append({
                    "start": start,
                    "end": end,
                    "bout_id": bout_id
                })

            else:
                parts = label.split("_")

                element_label = parts[0]
                confidence = float(parts[1])

                elements.append({
                    "start": start,
                    "end": end,
                    "type": element_label,
                    "confidence": confidence,
                })

    return elements, bouts


# -------------------------
# Assign elements to bouts
# -------------------------
def assign_elements_to_bouts(elements, bouts):
    bout_data = []

    for i, bout in enumerate(bouts):
        b_start, b_end = bout["start"], bout["end"]

        bout_elements = []
        for e in elements:
            # overlap condition
            if not (e["end"] < b_start or e["start"] > b_end):
                bout_elements.append(e)

        # sort by time
        bout_elements = sorted(bout_elements, key=lambda x: x["start"])

        bout_data.append({
            "bout_id": i,
            "start": b_start,
            "end": b_end,
            "elements": bout_elements
        })

    return bout_data


def load_audio(wav_path):
    audio, sr = sf.read(wav_path)

    if len(audio.shape) == 1:
        audio = audio[None, :]
    else:
        audio = audio.T

    return audio.astype(np.float32), sr

def waveform_to_logmel(
    waveform,
    sr=48000,
    n_mels=64,
    n_fft=1024,
    hop_length=144,   # 3 ms at 48kHz
    win_length=1024,
    f_min=50,
    f_max=20000,
    eps=1e-6
):
    """
    Convert waveform (1, T) → log-mel spectrogram (T_frames, n_mels)
    """

    # Mel spectrogram
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0
    )
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    mel_spec = mel_spec_transform(waveform)  # (1, n_mels, T)

    # Log scaling
    log_mel = torch.log(mel_spec + eps)

    # reshape to (T, F)
    log_mel = log_mel.squeeze(0).transpose(0, 1)

    return log_mel

def load_audio_segment(waveform, sr, start, end):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return waveform[:, start_sample:end_sample]


def process_file(wav_path, txt_path, output_dir):
    waveform, sr = load_audio(wav_path)

    assert sr == SR, f"Sample rate mismatch: {sr}"

    elements, bouts = parse_annotation_file(txt_path)
    bout_data = assign_elements_to_bouts(elements, bouts)

    base_name = Path(wav_path).stem

    stem = base_name.replace("_labels.txt", "")
    # ============================================================
    # PARSE SPEAKER INFORMATION
    # ============================================================

    parts = stem.split("_")

    speaker = parts[0]
    take_id = parts[1]

    label = SPEAKER_TO_ID[speaker]

    for bout in bout_data:

        element_images = []
        element_meta = []

        for e in bout["elements"]:
            ew = load_audio_segment(waveform, sr, e["start"], e["end"])

            mel = waveform_to_logmel(ew, sr)
            element_images.append(mel)

            element_meta.append({
                "type": e["type"],
                "start": e["start"],
                "end": e["end"],
                "confidence": e["confidence"],
                "duration": e["end"] - e["start"]
            })

        sample = {
            "elements": element_images,
            "lengths": [x.shape[0] for x in element_images],
            #"elements_meta": element_meta,
            "label": label,
            #"animal_id": speaker
        }

        # --- include animal in filename ---
        out_path = os.path.join(
            output_dir,
            f"{speaker}_{base_name}_bout_{bout['bout_id']}.pt"
        )

        torch.save(sample, out_path)

def process_dataset(wav_dir, pred_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

    for wav_file in wav_files:

        base = Path(wav_file).stem

        wav_path = os.path.join(wav_dir, wav_file)
        txt_path = os.path.join(pred_dir, f"{base}_labels.txt")

        if not os.path.exists(txt_path):
            print(f"Missing labels for {wav_file}")
            continue

        print(f"Processing {wav_file}")
        process_file(wav_path, txt_path, output_dir)

if __name__ == "__main__":
    GT_DIR = "D:/0_PHD/Hyrax/SpeakerIdentification/VLAD214/DATABASE/GROUND-TRUTH_AUDACITY/"
    ANNOTATION_DIR = "D:/0_PHD/Hyrax/SpeakerIdentification/VLAD214/DATABASE/Pred/"
    AUDIO_DIR = "D:/0_PHD/Hyrax/SpeakerIdentification/VLAD214/DATABASE/ACA/"
    OUTPUT_DIR = "D:/0_PHD/Hyrax/SpeakerIdentification/VLAD214/DATABASE/NPZ_pcen/"

    wav_dir = "C:/PHD/SideProjects/Vlad214/DATABASE/ORIGINAL/"
    pred_dir = "C:/PHD/SideProjects/Vlad214/DATABASE/Pred_GB/"
    output_dir = "C:/PHD/SideProjects/Vlad214/DATABASE/PT_Image_ORG_GB/"

    process_dataset(wav_dir, pred_dir, output_dir)