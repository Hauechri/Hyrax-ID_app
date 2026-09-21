import os
import joblib
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from dataset import (
    load_split,
    FusionPreprocessor,
)

from model import create_model


# =========================================================
# TRAIN
# =========================================================

def train(
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\nCreating model...")

    model = create_model()

    print("Training model...")

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_val, y_val)
        ],

        verbose=True
    )

    print("\nValidation Evaluation:\n")

    preds = model.predict(X_val)

    print(
        classification_report(
            y_val,
            preds
        )
    )

    return model


# =========================================================
# MAIN
# =========================================================

def main(
    data_folder,
    model_out
):

    print("\n============================")
    print(" FUSION TRAINING PIPELINE ")
    print("============================\n")

    # =====================================================
    # LOAD DATA
    # =====================================================

    train_folder = os.path.join(
        data_folder,
        "train"
    )

    val_folder = os.path.join(
        data_folder,
        "val"
    )

    test_folder = os.path.join(
        data_folder,
        "test"
    )

    print("Loading datasets...")

    (
        Xc_train,
        Xe_train,
        y_train,
        feature_keys
    ) = load_split(train_folder)

    (
        Xc_val,
        Xe_val,
        y_val,
        _
    ) = load_split(val_folder)

    (
        Xc_test,
        Xe_test,
        y_test,
        _
    ) = load_split(test_folder)

    # =====================================================
    # STATS
    # =====================================================

    print("\nDataset stats:")

    print(
        f"Train samples: {len(y_train)}"
    )

    print(
        f"Val samples:   {len(y_val)}"
    )

    print(
        f"Test samples:  {len(y_test)}"
    )

    print("\nKEEP ratio:")

    print(
        f"Train: {np.mean(y_train):.3f}"
    )

    print(
        f"Val:   {np.mean(y_val):.3f}"
    )

    # =====================================================
    # PREPROCESSOR
    # =====================================================

    print("\nFitting fusion preprocessor...")

    preprocessor = (
        FusionPreprocessor(
            pca_dim=64
        )
    )

    X_train = (
        preprocessor.fit_transform(
            Xc_train,
            Xe_train
        )
    )

    X_val = (
        preprocessor.transform(
            Xc_val,
            Xe_val
        )
    )

    X_test = (
        preprocessor.transform(
            Xc_test,
            Xe_test
        )
    )

    print("\nFused feature shape:")
    print(X_train.shape)

    # =====================================================
    # TRAIN
    # =====================================================

    model = train(
        X_train,
        y_train,
        X_val,
        y_val
    )

    # =====================================================
    # TEST
    # =====================================================

    print("\n============================")
    print(" FINAL TEST EVALUATION ")
    print("============================\n")

    test_preds = model.predict(
        X_test
    )

    print("Classification Report:\n")

    print(
        classification_report(
            y_test,
            test_preds
        )
    )

    print("Confusion Matrix:\n")

    cm = confusion_matrix(
        y_test,
        test_preds
    )

    print(cm)

    # =====================================================
    # SAVE
    # =====================================================

    print("\nSaving model...")

    save_data = {

        "model": model,

        "preprocessor": preprocessor,

        "feature_keys": feature_keys,
    }

    joblib.dump(
        save_data,
        model_out
    )

    print(
        f"Model saved to:\n{model_out}"
    )

    print("\nDone.")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    data_folder = r"D:/0_PHD/Hyrax/BoutGarbageCollector/DATA/ACA/GoodTargetGarbage/FusionDATASET/"

    model_out = r"D:/0_PHD/Hyrax/BoutGarbageCollector/DATA/ACA/GoodTargetGarbage/GarbageCollectorModel/Fusion/XGBRich.joblib"

    main(
        data_folder,
        model_out
    )