import re
import shutil
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_recording_name(filename: str) -> str:
    """
    Removes the '_bout_<number>.pt' suffix.

    Example:
        J9_J9_Take119_J9_300408_bout_12.pt
        -> J9_J9_Take119_J9_300408
    """
    match = re.match(r"(.+)_bout_\d+\.pt$", filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return match.group(1)


def get_animal_id(recording_name: str) -> str:
    """Animal/class is assumed to be the first underscore-separated part."""
    return recording_name.split("_")[0]


def recording_time_key(recording_name: str):
    """
    Sort priority:
      1. Valid 6-digit DDMMYY date
      2. Take<number>
      3. No usable information: placed last

    Returns a sortable tuple.
    """
    # Find standalone 6-digit number tokens, such as _300408_
    date_matches = re.findall(r"(?<!\d)(\d{6})(?!\d)", recording_name)

    valid_dates = []
    for date_text in date_matches:
        try:
            valid_dates.append(datetime.strptime(date_text, "%d%m%y"))
        except ValueError:
            pass

    if valid_dates:
        # Category 0 = dated recordings, ordered from oldest to newest
        return (0, min(valid_dates), 0, recording_name)

    take_match = re.search(r"Take[_-]?(\d+)", recording_name, re.IGNORECASE)
    if take_match:
        # Category 1 = recordings with a Take number
        return (1, datetime.max, int(take_match.group(1)), recording_name)

    # Category 2 = unknown chronology, deliberately placed last
    return (2, datetime.max, float("inf"), recording_name)


def collect_recordings(input_dir: Path):
    """
    Returns:
        animal -> recording_name -> [bout files]
    """
    recordings = defaultdict(lambda: defaultdict(list))

    for file_path in input_dir.glob("*.pt"):
        recording_name = parse_recording_name(file_path.name)
        animal = get_animal_id(recording_name)
        recordings[animal][recording_name].append(file_path)

    return recordings


def find_best_chronological_split(recordings, train_ratio=0.70, val_ratio=0.15):
    """
    recordings must already be chronologically sorted.

    Finds the best two boundaries:
        [oldest recordings] -> train
        [middle recordings] -> val
        [newest recordings] -> test

    Boundaries are selected to minimize deviation from target BOUT counts,
    while never splitting individual recordings.
    """
    n_recordings = len(recordings)

    if n_recordings == 1:
        return recordings, [], []

    if n_recordings == 2:
        return [recordings[0]], [], [recordings[1]]

    bout_counts = [len(files) for _, files in recordings]
    total_bouts = sum(bout_counts)

    target_train = total_bouts * train_ratio
    target_val = total_bouts * val_ratio
    target_test = total_bouts * (1 - train_ratio - val_ratio)

    prefix = [0]
    for count in bout_counts:
        prefix.append(prefix[-1] + count)

    best_score = float("inf")
    best_boundaries = None

    # train must receive at least one recording;
    # test must receive at least one recording.
    # Val is allowed to be empty if only two recordings exist,
    # but n_recordings >= 3 here, so all three receive one.
    for train_end in range(1, n_recordings - 1):
        for val_end in range(train_end + 1, n_recordings):
            train_bouts = prefix[train_end]
            val_bouts = prefix[val_end] - prefix[train_end]
            test_bouts = prefix[-1] - prefix[val_end]

            # Squared error favors a close overall 70/15/15 allocation.
            score = (
                (train_bouts - target_train) ** 2
                + (val_bouts - target_val) ** 2
                + (test_bouts - target_test) ** 2
            )

            if score < best_score:
                best_score = score
                best_boundaries = (train_end, val_end)

    train_end, val_end = best_boundaries

    train = recordings[:train_end]
    val = recordings[train_end:val_end]
    test = recordings[val_end:]

    return train, val, test


def flatten_files(recording_groups):
    """Converts [(recording_name, [files]), ...] into one file list."""
    return [
        file_path
        for _, files in recording_groups
        for file_path in files
    ]


def copy_files(files, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        output_path = destination / file_path.name

        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {output_path}"
            )

        shutil.copy2(file_path, output_path)


def split_dataset(input_dir, output_dir, selected_animals=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    recordings_by_animal = collect_recordings(input_dir)

    summary = defaultdict(int)

    for animal in sorted(recordings_by_animal):
        if selected_animals and animal not in selected_animals:
            continue

        recording_map = recordings_by_animal[animal]

        chronological_recordings = sorted(
            recording_map.items(),
            key=lambda item: recording_time_key(item[0])
        )

        train_groups, val_groups, test_groups = find_best_chronological_split(
            chronological_recordings
        )

        train_files = flatten_files(train_groups)
        val_files = flatten_files(val_groups)
        test_files = flatten_files(test_groups)

        copy_files(train_files, output_dir / "train")
        copy_files(val_files, output_dir / "val")
        copy_files(test_files, output_dir / "test")

        total = len(train_files) + len(val_files) + len(test_files)

        summary["animals"] += 1
        summary["recordings"] += len(chronological_recordings)
        summary["total_bouts"] += total
        summary["train_bouts"] += len(train_files)
        summary["val_bouts"] += len(val_files)
        summary["test_bouts"] += len(test_files)

        print(f"\n{animal}")
        print(f"  Recordings: {len(chronological_recordings)}")
        print(
            f"  Bouts: total={total} | "
            f"train={len(train_files)} ({len(train_files) / total:.1%}) | "
            f"val={len(val_files)} ({len(val_files) / total:.1%}) | "
            f"test={len(test_files)} ({len(test_files) / total:.1%})"
        )

        print("  Train recordings:", [name for name, _ in train_groups])
        print("  Val recordings:  ", [name for name, _ in val_groups])
        print("  Test recordings: ", [name for name, _ in test_groups])

    total_bouts = summary["total_bouts"]

    print("\n==============================")
    print("FINAL SUMMARY")
    print("==============================")
    print(f"Animals: {summary['animals']}")
    print(f"Recordings: {summary['recordings']}")
    print(f"Total bouts: {total_bouts}")

    if total_bouts:
        print(
            f"Train: {summary['train_bouts']} "
            f"({summary['train_bouts'] / total_bouts:.1%})"
        )
        print(
            f"Val:   {summary['val_bouts']} "
            f"({summary['val_bouts'] / total_bouts:.1%})"
        )
        print(
            f"Test:  {summary['test_bouts']} "
            f"({summary['test_bouts'] / total_bouts:.1%})"
        )


if __name__ == "__main__":
    selected_animals = {
        "J9", "Kashtan", "M0", "M9", "O1", "O7",
        "P0", "P1", "P8", "Q7", "R3", "T0",
        "T1", "T9", "U7", "U9", "W4", "X0",
    }

    input_dir = r"C:/PHD/SideProjects/Vlad214/DATABASE/TEST_PT_ECAPATDNN_ORG_GB_96/"
    output_dir = r"C:/PHD/SideProjects/Vlad214/DATASET/TEST_FileAge_ECAPATDNN_96/"

    split_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        selected_animals=selected_animals,
    )