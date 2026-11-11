"""Audacity label output: the writer (extracted unchanged) and the reader used by the spectrogram viewer.

No heavy imports on purpose: the viewer's batch worker processes import this module."""


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

    with open(output_path, "w", encoding="utf-8") as f:   # parse_bout_results reads utf-8
        for start, end, label in lines:
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")


def parse_bout_results(label_path):
    """
    Parse a *_labels.txt file written by run_pipeline() into structured bout
    dicts: [{"bout_id", "start", "end", "animal", "confidence", "elements": [...]}, ...]

    If the run had classification disabled, bout headers are just "BOUT{n}"
    with no animal/confidence; those bouts come back with animal=None and
    confidence=None.

    NOTE: HYRAX_ID_predict.py's own parse_annotation_file()/assign_elements_to_bouts()
    target an older ground-truth label format ("BOUT_1", no animal/confidence
    suffix) and raise ValueError on the pipeline's actual output format
    ("BOUT1_{animal}_{conf}") -- confirmed by inspection, not assumption.
    This parses the real output format instead.

    Relies on save_full_audacity_labels() always writing each bout's element
    lines immediately after that bout's own header line (true because a
    bout's start time ties with its first element's start time, and Python's
    stable sort preserves the original append order on ties) -- verified
    against real pipeline output.
    """
    bouts = []
    current = None
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                start_s, end_s, label = line.split("\t")
                start, end = float(start_s), float(end_s)
                if label.startswith("BOUT"):
                    # "BOUT{n}_{animal}_{conf}": the animal name may itself contain underscores, so the
                    # id is split off from the left and the confidence from the right.
                    parts = label[len("BOUT"):].split("_", 1)
                    if len(parts) == 2:
                        bout_id_s = parts[0]
                        animal, conf_s = parts[1].rsplit("_", 1)
                        confidence = float(conf_s)
                    else:
                        # classification was disabled: bare "BOUT{n}"
                        bout_id_s, animal, confidence = parts[0], None, None
                    current = {
                        "bout_id": int(bout_id_s),
                        "start": start,
                        "end": end,
                        "animal": animal,
                        "confidence": confidence,
                        "elements": [],
                    }
                    bouts.append(current)
                elif current is not None:
                    elem_type, conf_s = label.rsplit("_", 1)
                    current["elements"].append(
                        {"start": start, "end": end, "type": elem_type, "confidence": float(conf_s)}
                    )
            except ValueError:
                # Defensive only: a line that doesn't match this run's own
                # output format (e.g. this file wasn't actually written by
                # save_full_audacity_labels). list_input_stems() is the real
                # fix for not reading stray files in the first place; this
                # just keeps one bad line from losing the whole file's results.
                continue
    return bouts
