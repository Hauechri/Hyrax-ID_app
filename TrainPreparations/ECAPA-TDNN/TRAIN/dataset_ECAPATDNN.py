import os
import torch
from torch.utils.data import Dataset


class BoutDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ]

        self.files.sort()

    def __len__(self):
        return len(self.files)

    def _validate_element(self, e, path):

        if not torch.is_tensor(e):
            e = torch.tensor(e)

        if e.ndim != 2:
            raise ValueError(
                f"[{path}] Bad shape {e.shape}, expected (T,F)"
            )

        return e.float()

    def __getitem__(self, idx):
        path = self.files[idx]
        sample = torch.load(path)

        # -------------------------
        # Required fields check
        # -------------------------
        if "elements" not in sample:
            raise KeyError(f"{path} missing 'elements'")

        if "label" not in sample:
            raise KeyError(f"{path} missing 'label'")

        raw_elements = sample["elements"]
        label = sample["label"]

        # -------------------------
        # Clean elements
        # -------------------------
        elements = [
            self._validate_element(e, path)
            for e in raw_elements
        ]

        return {
            "elements": elements,   # list of (T, 64)
            "label": torch.tensor(label, dtype=torch.long),
            "file": os.path.basename(path)
        }


# -------------------------
# Collate function
# -------------------------
def collate_fn(batch):
    """
    Keeps variable-length structure intact.
    """

    return {
        "elements": [item["elements"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch]),
        "files": [item["file"] for item in batch]
    }