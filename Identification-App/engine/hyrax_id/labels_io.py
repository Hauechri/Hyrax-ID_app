"""Audacity label output (extracted unchanged)."""


def save_full_audacity_labels(bout_data, output_path):
    """
    Writes one Audacity label file containing, per bout:
      - a bout-level label:    start / end / "BOUT{n}_{animal}_{conf:.3f}"
                               (or just "BOUT{n}" if the bout was not classified,
                               i.e. it has no "animal" key)
      - element-level labels:  start / end / "{type}_{conf:.3f}"
    """
    lines = []
    for bout in bout_data:
        if "animal" in bout:
            bout_label = f"BOUT{bout['bout_id']}_{bout['animal']}_{bout['animal_confidence']:.3f}"
        else:
            bout_label = f"BOUT{bout['bout_id']}"
        lines.append((bout["start"], bout["end"], bout_label))
        for e in bout["elements"]:
            lines.append((e["start"], e["end"], f"{e['type']}_{e['confidence']:.3f}"))

    lines.sort(key=lambda x: x[0])

    with open(output_path, "w") as f:
        for start, end, label in lines:
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")
