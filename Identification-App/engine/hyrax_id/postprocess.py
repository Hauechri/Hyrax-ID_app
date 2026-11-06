"""Label merging, bout grouping and bout assembly (extracted unchanged)."""
from collections import defaultdict


def merge_overlapping_labels(label_list, overlap_threshold=0.5, min_fraction_keep=0.5):
    """
    Merge labels of the same class if they overlap by at least a given fraction,
    then resolve overlaps between different classes based on confidence.
    """
    def overlap_fraction(a_start, a_end, b_start, b_end):
        overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
        len_a, len_b = a_end - a_start, b_end - b_start
        denom = min(len_a, len_b)
        return overlap / denom if denom > 0 else 0

    class_groups = defaultdict(list)
    for start, end, cls, conf in label_list:
        class_groups[cls].append((start, end, conf))

    merged_labels = []
    for cls, intervals in class_groups.items():
        intervals.sort()
        current_start, current_end, current_confs = intervals[0][0], intervals[0][1], [intervals[0][2]]

        for start, end, conf in intervals[1:]:
            frac = overlap_fraction(current_start, current_end, start, end)
            if frac >= overlap_threshold:
                current_end = max(current_end, end)
                current_confs.append(conf)
            else:
                mean_conf = sum(current_confs) / len(current_confs)
                merged_labels.append((current_start, current_end, cls, mean_conf))
                current_start, current_end, current_confs = start, end, [conf]

        mean_conf = sum(current_confs) / len(current_confs)
        merged_labels.append((current_start, current_end, cls, mean_conf))

    merged_labels.sort(key=lambda x: x[0])

    final_labels = []
    for label in merged_labels:
        start, end, cls, conf = label
        keep = True
        for i, (s2, e2, c2, conf2) in enumerate(final_labels):
            if e2 <= start or s2 >= end:
                continue

            overlap_start = max(start, s2)
            overlap_end = min(end, e2)
            overlap_len = overlap_end - overlap_start
            if overlap_len <= 0:
                continue

            if conf > conf2:
                orig_len = e2 - s2
                if overlap_start <= s2 and overlap_end >= e2:
                    final_labels[i] = None
                elif overlap_start <= s2:
                    new_start = overlap_end
                    new_len = e2 - new_start
                    final_labels[i] = (new_start, e2, c2, conf2) if new_len >= min_fraction_keep * orig_len else None
                elif overlap_end >= e2:
                    new_end = overlap_start
                    new_len = new_end - s2
                    final_labels[i] = (s2, new_end, c2, conf2) if new_len >= min_fraction_keep * orig_len else None
                else:
                    final_labels[i] = None
            else:
                orig_len = end - start
                if overlap_start <= start and overlap_end >= end:
                    keep = False
                    break
                elif overlap_start <= start:
                    start = overlap_end
                    new_len = end - start
                    if new_len < min_fraction_keep * orig_len:
                        keep = False
                        break
                elif overlap_end >= end:
                    end = overlap_start
                    new_len = end - start
                    if new_len < min_fraction_keep * orig_len:
                        keep = False
                        break
                else:
                    keep = False
                    break

        final_labels = [lbl for lbl in final_labels if lbl is not None]
        if keep:
            final_labels.append((start, end, cls, conf))

    final_labels.sort(key=lambda x: x[0])
    return final_labels


def group_bouts(merged_labels, interarrival_threshold=0.724):
    """Group merged labels into bouts based on interarrival time.

    A label with cls == 2 always starts a new bout.
    """
    if not merged_labels:
        return []

    merged_labels = sorted(merged_labels, key=lambda x: x[0])

    bouts = []
    bout_start = merged_labels[0][0]
    bout_end = merged_labels[0][1]
    bout_number = 1

    for i in range(1, len(merged_labels)):
        start, end, cls, conf = merged_labels[i]

        gap = start - bout_end

        if cls == 2 or gap > interarrival_threshold:
            # Finish the current bout
            bouts.append(
                (bout_start, bout_end, "BOUT", float(bout_number))
            )
            bout_number += 1

            # Start a new bout with the current label
            bout_start = start
            bout_end = end
        else:
            # Continue the current bout
            bout_end = max(bout_end, end)

    # Append the final bout
    bouts.append(
        (bout_start, bout_end, "BOUT", float(bout_number))
    )

    return bouts


def convert_to_detections(merged_labels):
    detections = []
    for i, (start, end, cls, conf) in enumerate(merged_labels):
        detections.append({"start": start, "end": end, "conf": conf, "class_id": cls, "idx": i})
    return detections


def build_bout_data(elements, bouts):
    """
    elements: list of (start, end, cls_name, conf)
    bouts:    list of (start, end, "BOUT", bout_number)

    Returns a list of dicts:
        {bout_id, start, end, elements: [{start, end, type, confidence}, ...]}
    """
    bout_data = []

    for b_start, b_end, _, bout_number in bouts:
        bout_elements = []
        for start, end, cls, conf in elements:
            # overlap condition
            if not (end < b_start or start > b_end):
                bout_elements.append({
                    "start": start,
                    "end": end,
                    "type": cls,
                    "confidence": conf,
                })

        bout_elements.sort(key=lambda x: x["start"])

        bout_data.append({
            "bout_id": int(bout_number),
            "start": b_start,
            "end": b_end,
            "elements": bout_elements,
        })

    return bout_data
