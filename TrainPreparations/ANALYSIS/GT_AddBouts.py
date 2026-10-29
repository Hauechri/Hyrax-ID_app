import os

class_map = {0: "chuck", 1: "snort", 2: "wail"}
valid_labels = set(class_map.values())

threshold = 0.724 # seconds


def load_elements(file_path):
    elements = []
    preserved_lines = []

    with open(file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            parts = stripped.split()

            if len(parts) < 3:
                preserved_lines.append(line.rstrip())
                continue

            try:
                start = float(parts[0].replace(",", "."))
                end = float(parts[1].replace(",", "."))
            except ValueError:
                preserved_lines.append(line.rstrip())
                continue

            label = parts[2].lower()

            # Remove old bout labels
            if label.startswith("bout"):
                continue

            # Keep valid annotations for processing
            if label in valid_labels:
                elements.append({
                    "start": start,
                    "end": end,
                    "label": label
                })

            # Preserve original non-bout lines
            preserved_lines.append(line.rstrip())

    elements.sort(key=lambda x: x["start"])
    return elements, preserved_lines


def group_bouts(elements, threshold):

    if not elements:
        return []

    bouts = []
    current_bout = [elements[0]]

    for prev, curr in zip(elements[:-1], elements[1:]):

        interarrival = curr["start"] - prev["end"]

        if interarrival <= threshold:
            current_bout.append(curr)
        else:
            bouts.append(current_bout)
            current_bout = [curr]

    bouts.append(current_bout)

    return bouts


def write_output(original_lines, bouts, output_path):

    with open(output_path, "w") as f:

        # write original labels
        for line in original_lines:
            f.write(line + "\n")

        # append bouts
        for i, bout in enumerate(bouts):

            start = bout[0]["start"]
            end = bout[-1]["end"]

            f.write(f"{start:.6f}\t{end:.6f}\tbout_{i}\n")


def process_folder(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):

        if not file.endswith(".txt"):
            continue

        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        elements, original_lines = load_elements(input_path)

        bouts = group_bouts(elements, threshold)

        write_output(original_lines, bouts, output_path)

        print(f"Processed {file} → {len(bouts)} bouts")


if __name__ == "__main__":

    input_folder = "C:/PHD/SideProjectsData/HeloiseNewDownload/male_songs_combined_paired/Labels/"
    output_folder = "C:/PHD/SideProjectsData/HeloiseNewDownload/male_songs_combined_paired/Labels_BT/"

    process_folder(input_folder, output_folder)