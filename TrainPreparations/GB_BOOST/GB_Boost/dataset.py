import os
import numpy as np


class GarbageDataset:

    def __init__(self, folder):

        self.folder = folder

        self.files = sorted([
            f for f in os.listdir(folder)
            if f.endswith(".npz")
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No npz files found in: {folder}")

    def load(self):

        X_all = []
        y_all = []

        feature_keys = None

        print(f"\nLoading dataset from: {self.folder}")

        total_samples = 0

        for fname in self.files:

            path = os.path.join(self.folder, fname)

            data = np.load(path, allow_pickle=True)

            X = data["X"]
            y = data["y"]

            keys = data["feature_keys"]

            if feature_keys is None:
                feature_keys = keys

            X_all.append(X)
            y_all.append(y)

            print(f"{fname}: {len(y)} samples")

            total_samples += len(y)

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        print("\n==============================")
        print(f"Total files: {len(self.files)}")
        print(f"Total samples: {total_samples}")
        print("==============================")

        return X_all, y_all, feature_keys