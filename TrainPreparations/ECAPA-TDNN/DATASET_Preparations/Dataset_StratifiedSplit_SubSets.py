import os
import shutil
import random
from collections import defaultdict
from pathlib import Path
import argparse


def parse_base_name(filename: str) -> str:
    """
    Extract base filename up to '_bout'
    Example:
        R1_240504_0025_bout_0.npy/pt -> R1_240504_0025
    """
    if "_bout" not in filename:
        raise ValueError(f"Filename does not contain '_bout': {filename}")
    return filename.split("_bout")[0]


def collect_files(input_dir: Path):
    groups = defaultdict(list)

    for f in input_dir.glob("*.pt"):
        base = parse_base_name(f.name)
        groups[base].append(f)

    return groups


def split_group(files, train_ratio=0.7, val_ratio=0.15, seed=42):
    files = list(files)
    random.Random(seed).shuffle(files)

    n = len(files)

    # Special handling for tiny groups
    if n == 1:
        return files, [], []

    if n == 2:
        return [files[0]], [], [files[1]]

    if n == 3:
        return [files[0]], [files[1]], [files[2]]

    # Desired floating point allocations
    raw_train = n * train_ratio
    raw_val = n * val_ratio
    raw_test = n * (1 - train_ratio - val_ratio)

    # Floor first
    n_train = int(raw_train)
    n_val = int(raw_val)
    n_test = int(raw_test)

    allocated = n_train + n_val + n_test
    remaining = n - allocated

    # Largest remainder method
    remainders = [
        ("train", raw_train - n_train),
        ("val", raw_val - n_val),
        ("test", raw_test - n_test),
    ]

    remainders.sort(key=lambda x: x[1], reverse=True)

    for i in range(remaining):
        split_name = remainders[i][0]

        if split_name == "train":
            n_train += 1
        elif split_name == "val":
            n_val += 1
        else:
            n_test += 1

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    return train, val, test


def copy_files(file_list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in file_list:
        shutil.copy2(f, out_dir / f.name)


def main(input_dir, output_dir, seed=42):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    groups = collect_files(input_dir)

    summary = {
        "total_groups": len(groups),
        "total_files": 0,
        "train": 0,
        "val": 0,
        "test": 0,
    }

    print("\n📊 Dataset Statistics per Base Filename:\n")

    for base, files in groups.items():

        group_id = base.split("_")[0]

        if group_id not in SelectedGroups:
            continue

        train, val, test = split_group(files, seed=seed)

        summary["total_files"] += len(files)
        summary["train"] += len(train)
        summary["val"] += len(val)
        summary["test"] += len(test)

        print(f"🔹 {base}")
        print(f"   total bouts: {len(files)}")
        print(f"   train: {len(train)} | val: {len(val)} | test: {len(test)}")

        copy_files(train, output_dir / "train")
        copy_files(val, output_dir / "val")
        copy_files(test, output_dir / "test")

    print("\n==============================")
    print("FINAL SUMMARY")
    print("==============================")
    print(f"Groups: {summary['total_groups']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Train: {summary['train']}")
    print(f"Val: {summary['val']}")
    print(f"Test: {summary['test']}")


if __name__ == "__main__":
    SelectedGroups = ['J9', 'Kashtan', 'M0', 'M9', 'O1', 'O7', 'P0', 'P1', 'P8', 'Q7', 'R3', 'T0', 'T1', 'T9', 'U7', 'U9', 'W4', 'X0']
    input_dir = "C:/PHD/SideProjects/Vlad214/DATABASE/TEST_PT_ECAPATDNN_ORG_GB_96/"
    output_dir = "C:/PHD/SideProjects/Vlad214/DATASET/TEST_Stratified_ECAPATDNN_96/"
    seed = 7324562398634

    main(input_dir, output_dir, seed)