#DEPRECATED

import os
import glob
import soundfile as sf


class Element:
    def __init__(self, start, end, label, bout_id, conf=1.0):
        self.start = start
        self.end = end
        self.label = label
        self.bout_id = bout_id
        self.conf = conf


def parse_file(filepath):
    elements = []

    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            start = float(parts[0].replace(",", "."))
            end = float(parts[1].replace(",", "."))
            label = parts[2].lower()

            if "bout" in label:
                continue

            elements.append(Element(start, end, label, bout_id=-1))

    return elements


def extract_snippet(audio, sr, start, end, pad=0.2):
    start = max(0, start - pad)
    end = end + pad

    return audio[int(start * sr):int(end * sr)]


def main(gt_folder, audio_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for gt_file in glob.glob(os.path.join(gt_folder, "*.txt")):
        base = os.path.basename(gt_file).replace(".txt", "")
        audio_file = os.path.join(audio_folder, base + ".wav")

        if not os.path.exists(audio_file):
            continue

        elements = parse_file(gt_file)
        audio, sr = sf.read(audio_file)

        for i, elem in enumerate(elements):
            snippet = extract_snippet(audio, sr, elem.start, elem.end)

            out_path = os.path.join(
                output_folder,
                f"{base}_GT_{i}_{elem.label}.wav"
            )

            sf.write(out_path, snippet, sr)

    print("Done extracting GT-only dataset.")

#DEPRECATED
if __name__ == "__main__":
    GT_FOLDER = "D:/0_PHD/Hyrax/ImprovedAnalysis/AdditionalValidation/AutomaticBoutAudacity/"
    AUDIO_FOLDER = "D:/0_PHD/Hyrax/ImprovedAnalysis/AdditionalValidation/AUDIO/ANIMAL-CLEAN-ADAPT/"
    OUTPUT = "D:/0_PHD/Hyrax/BoutGarbageCollector/AUDIO/ACA/Additional/Target/"

    main(GT_FOLDER, AUDIO_FOLDER, OUTPUT)