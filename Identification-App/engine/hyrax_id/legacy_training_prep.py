"""Reference only -- NOT used by inference. Training-data prep helpers (extracted unchanged)."""
from pathlib import Path

from .audio_io import load_audio, load_audio_segment
from .classifier import waveform_to_logmel
from .config import SR


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
                bouts.append({"start": start, "end": end, "bout_id": bout_id})
            else:
                parts = label.split("_")
                elements.append({
                    "start": start,
                    "end": end,
                    "type": parts[0],
                    "confidence": float(parts[1]),
                })

    return elements, bouts


def assign_elements_to_bouts(elements, bouts):
    bout_data = []
    for i, bout in enumerate(bouts):
        b_start, b_end = bout["start"], bout["end"]
        bout_elements = [e for e in elements if not (e["end"] < b_start or e["start"] > b_end)]
        bout_elements = sorted(bout_elements, key=lambda x: x["start"])
        bout_data.append({"bout_id": i, "start": b_start, "end": b_end, "elements": bout_elements})
    return bout_data


def process_file(wav_path, txt_path, speaker_to_id):
    """Training-data prep helper. Not used for inference."""
    waveform, sr = load_audio(wav_path)
    assert sr == SR, f"Sample rate mismatch: {sr}"

    elements, bouts = parse_annotation_file(txt_path)
    bout_data = assign_elements_to_bouts(elements, bouts)

    base_name = Path(wav_path).stem
    stem = base_name.replace("_labels.txt", "")
    parts = stem.split("_")
    speaker = parts[0]

    label = speaker_to_id[speaker]

    for bout in bout_data:
        element_images = [
            waveform_to_logmel(load_audio_segment(waveform, sr, e["start"], e["end"]), sr)
            for e in bout["elements"]
        ]
        sample = {
            "elements": element_images,
            "lengths": [x.shape[0] for x in element_images],
            "label": label,
        }
        return sample
