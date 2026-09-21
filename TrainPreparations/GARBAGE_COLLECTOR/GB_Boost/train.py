import os
import joblib
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from dataset import GarbageDataset
from model import create_model


# =========================================================
# TRAIN
# =========================================================

def train_model(X_train, y_train):

    print("\nCreating model...")
    model = create_model()

    print("Training model...")
    model.fit(X_train, y_train)

    return model


# =========================================================
# EVALUATE
# =========================================================

def evaluate_model(model, X, y, name="Evaluation"):

    print("\n============================")
    print(f" {name.upper()} ")
    print("============================\n")

    preds = model.predict(X)

    print("Classification Report:\n")
    print(classification_report(y, preds))

    print("Confusion Matrix:\n")

    cm = confusion_matrix(y, preds)

    print(cm)


# =========================================================
# MAIN
# =========================================================

def main(data_folder, model_out):

    print("\n============================")
    print(" GC TRAINING PIPELINE START ")
    print("============================")

    # -----------------------------------------------------
    # DATASET PATHS
    # -----------------------------------------------------

    train_folder = os.path.join(data_folder, "train")
    val_folder = os.path.join(data_folder, "val")
    test_folder = os.path.join(data_folder, "test")

    # -----------------------------------------------------
    # LOAD DATASETS
    # -----------------------------------------------------

    train_dataset = GarbageDataset(train_folder)
    val_dataset = GarbageDataset(val_folder)
    test_dataset = GarbageDataset(test_folder)

    X_train, y_train, feature_keys = train_dataset.load()
    X_val, y_val, _ = val_dataset.load()
    X_test, y_test, _ = test_dataset.load()

    # -----------------------------------------------------
    # DATASET STATS
    # -----------------------------------------------------

    print("\n============================")
    print(" DATASET STATS ")
    print("============================")

    print(f"\nTrain samples: {len(y_train)}")
    print(f"Val samples:   {len(y_val)}")
    print(f"Test samples:  {len(y_test)}")

    print("\nKEEP ratio:")

    print(f"Train: {np.mean(y_train):.3f}")
    print(f"Val:   {np.mean(y_val):.3f}")
    print(f"Test:  {np.mean(y_test):.3f}")

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model = train_model(X_train, y_train)

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    evaluate_model(
        model,
        X_val,
        y_val,
        name="Validation Evaluation"
    )

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test,
        name="Final Test Evaluation"
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    print("\nSaving model...")

    save_data = {
        "model": model,
        "feature_keys": feature_keys,
    }

    joblib.dump(save_data, model_out)

    print(f"Saved model to: {model_out}")

    print("\nDone.")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    data_folder = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/BoostTrain/BoostDataset/"

    model_out = r"C:/PHD/SideProjectsData/IcasspExperiments/YOLOExperiments/TrainingData/GB_Boost_Model/XGBRich.joblib"

    main(data_folder, model_out)