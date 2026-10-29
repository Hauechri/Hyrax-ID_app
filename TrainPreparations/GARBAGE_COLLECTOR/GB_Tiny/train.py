import os
import argparse
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from model import create_model


# =========================
# LOAD DATA
# =========================

def load_npz(path):
    data = np.load(path)
    return data["X"], data["y"]


# =========================
# TRAIN FUNCTION
# =========================

def train(X_train, y_train, X_val, y_val):
    print("\nCreating model...")
    model = create_model()

    print("Training model...")
    model.fit(X_train, y_train)

    print("\nEvaluating...")
    preds = model.predict(X_val)

    print(classification_report(y_val, preds))

    return model


# =========================
# MAIN
# =========================

def main(data_folder, model_out):

    print("\n============================")
    print(" GC TRAINING PIPELINE START ")
    print("============================\n")

    train_path = os.path.join(data_folder, "train.npz")
    val_path   = os.path.join(data_folder, "val.npz")
    test_path  = os.path.join(data_folder, "test.npz")

    print("Loading datasets...")

    X_train, y_train = load_npz(train_path)
    X_val, y_val     = load_npz(val_path)
    X_test, y_test   = load_npz(test_path)

    print("\nDataset stats:")
    print(f"  Train samples: {len(y_train)}")
    print(f"  Val samples:   {len(y_val)}")
    print(f"  Test samples:  {len(y_test)}")

    print(f"\nKEEP ratio:")
    print(f"  Train: {np.mean(y_train):.3f}")
    print(f"  Val:   {np.mean(y_val):.3f}")

    print("\nStarting training...\n")

    model = train(X_train, y_train, X_val, y_val)

    print("\n============================")
    print(" FINAL TEST EVALUATION ")
    print("============================\n")

    # Predictions
    test_preds = model.predict(X_test)

    # Classification report
    print("Classification Report:\n")
    print(classification_report(y_test, test_preds))

    # Confusion matrix
    print("Confusion Matrix:\n")
    cm = confusion_matrix(y_test, test_preds)
    print(cm)

    print("\nSaving model...")
    joblib.dump(model, model_out)

    print(f"Model saved to: {model_out}")

    print("\nDone.")



if __name__ == "__main__":
    data_folder = "D:/0_PHD/Hyrax/BoutGarbageCollector/AUDIO/ACA/GoodTargetGarbage/DATASET/"
    output = "D:/0_PHD/Hyrax/BoutGarbageCollector/AUDIO/ACA/GoodTargetGarbage/GarbageCollectorModel/GBTiny.joblib"

    main(data_folder, output)