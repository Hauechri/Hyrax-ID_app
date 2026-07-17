import os

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_tdnn import BoutDataset, collate_fn
from model_tdnn import TDNNBoutModel
from collections import Counter
from torch.utils.data import WeightedRandomSampler


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "C:/PHD/SideProjects/Vlad214/DATASET/7-13_David/train"
VAL_DIR   = "C:/PHD/SideProjects/Vlad214/DATASET/7-13_David/val"
TEST_DIR  = "C:/PHD/SideProjects/Vlad214/DATASET/7-13_David/test"

MODELOUTPUT = "C:/PHD/Hyrax-ID_app/HYRAX-ID_Inference/Tdnn/7-13_David_MHA/"

CHECKPOINT_PATH = os.path.join(MODELOUTPUT, "checkpoint.pt")
BEST_MODEL_PATH = os.path.join(MODELOUTPUT, "best_model.pt")

os.makedirs(MODELOUTPUT, exist_ok=True)

NUM_CLASSES = 18
class_list = sorted(
    ['J9', 'Kashtan', 'M0', 'M9', 'O1', 'O7', 'P0', 'P1', 'P8', 'Q7', 'R3', 'T0', 'T1', 'T9', 'U7', 'U9', 'W4', 'X0']
)

BATCH_SIZE = 1

EPOCHS = 150

LR = 1e-5  # or even 1e-5

weight_decay = 1e-6

PATIENCE = 15


# -------------------------
# Accuracy
# -------------------------
def compute_accuracy(logits, labels):

    preds = torch.argmax(logits, dim=1)

    return (preds == labels).float().mean().item()


# -------------------------
# Train
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0
    total_acc = 0

    for batch in tqdm(loader, desc="Train", leave=False):

        labels = batch["labels"].to(DEVICE)

        elements = [
            [e.to(DEVICE) for e in elements_list]
            for elements_list in batch["elements"]
        ]

        optimizer.zero_grad()

        logits = model(elements)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        acc = compute_accuracy(logits, labels)

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, name="Val", return_preds=False):

    model.eval()

    total_loss = 0
    total_acc = 0

    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc=name, leave=False):

        labels = batch["labels"].to(DEVICE)

        elements = [
            [e.to(DEVICE) for e in elements_list]
            for elements_list in batch["elements"]
        ]

        logits = model(elements)

        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        acc = (preds == labels).float().mean().item()

        total_loss += loss.item()
        total_acc += acc

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)

    if return_preds:
        return avg_loss, avg_acc, np.array(all_labels), np.array(all_preds)

    return avg_loss, avg_acc


# -------------------------
# Main
# -------------------------
def main():

    train_ds = BoutDataset(TRAIN_DIR)
    val_ds   = BoutDataset(VAL_DIR)
    test_ds  = BoutDataset(TEST_DIR)

    # ============================================================
    # WEIGHTEDRANDOMSAMPLING
    # ============================================================

    train_labels = []

    for f in train_ds.files:
        data = torch.load(f)
        train_labels.append(int(data["label"]))

    train_labels = np.array(train_labels)
    class_counts = Counter(train_labels)

    # weights = [
    #    1.0 / math.sqrt(class_counts[label])
    #    for label in train_labels
    # ]
    # Alternativ:
    weights = [
        1.0 / class_counts[label]
        for label in train_labels
    ]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,  # must NOT be True
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = TDNNBoutModel(
        num_classes=NUM_CLASSES,
        input_dim=64,
        emb_dim=128
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # -------------------------
    # Resume checkpoint
    # -------------------------
    if os.path.exists(CHECKPOINT_PATH):

        print("Loading checkpoint...")

        ckpt = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )

        model.load_state_dict(ckpt["model_state"])

        optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )

        start_epoch = ckpt["epoch"] + 1

        best_val_loss = ckpt["best_val_loss"]

        patience_counter = ckpt["patience_counter"]

        print(f"Resumed from epoch {start_epoch}")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            name="Val"
        )

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter
        }

        torch.save(checkpoint, CHECKPOINT_PATH)

        # -------------------------
        # Best model
        # -------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print("New best model saved")

        else:

            patience_counter += 1

            print(
                f"Patience: {patience_counter}/{PATIENCE}"
            )

        checkpoint["best_val_loss"] = best_val_loss
        checkpoint["patience_counter"] = patience_counter

        torch.save(checkpoint, CHECKPOINT_PATH)

        # -------------------------
        # Early stopping
        # -------------------------
        if patience_counter >= PATIENCE:

            print("Early stopping triggered")

            break

    # -------------------------
    # Test
    # -------------------------
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    )

    test_loss, test_acc, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        name="Test",
        return_preds=True
    )

    print("\n===== FINAL TEST =====")

    print(f"Test Loss: {test_loss:.4f}")

    print(f"Test Acc : {test_acc:.4f}")

    # -------------------------
    # Classification Report
    # -------------------------
    print("\n===== CLASSIFICATION REPORT =====")

    #['R1', 'SN', 'T5', 'T6']

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_list,
        digits=4,
        zero_division=0
    )

    print(report)

    # -------------------------
    # Confusion Matrix
    # -------------------------
    print("\n===== CONFUSION MATRIX =====")

    cm = confusion_matrix(y_true, y_pred)

    print(cm)


if __name__ == "__main__":
    main()