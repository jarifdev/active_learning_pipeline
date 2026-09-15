"""
train_random.py

Trains the RANDOM-baseline ResNet50 for one cycle.

It uses the same training setup as the baseline / active-learning branch,
but reads the growing random training set from:
    states/random/cycle_XX.json

and saves checkpoints to:
    checkpoints/random/cycle_XX.pt

Examples:
    python src/train_random.py --cycle 1
    python src/train_random.py --cycle 2
"""

from pathlib import Path
import argparse
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DefectDataset, ActiveLearningDataset
from transforms import get_train_transforms, get_eval_transforms
from model import build_model

MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
SPLIT_JSON = PROJECT_ROOT / "split.json"
DATA_DIR = PROJECT_ROOT / "raw_data"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 2
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        print("GPU detected -- training will use CUDA.")
        return torch.device("cuda")
    print("No GPU detected -- training will use CPU.")
    return torch.device("cpu")


def run_one_epoch(model, dataloader, criterion, optimizer, device, is_training):
    model.train() if is_training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True, help="Random baseline cycle: 1 to 5")
    parser.add_argument("--pipeline", choices=["old_pipeline", "new_pipeline"], required=True)
    args = parser.parse_args()
    cycle = args.cycle
    pipeline = args.pipeline
    STATES_DIR = PROJECT_ROOT / "states" / "resnet" / pipeline / "random"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "resnet" / pipeline / "random"
    LOG_DIR = PROJECT_ROOT / "logs" / "resnet" / pipeline / "random"

    if cycle < 1 or cycle > 5:
        raise ValueError("Cycle must be between 1 and 5.")

    set_seed(SEED)

    state_path = STATES_DIR / f"cycle_{cycle:02d}.json"
    checkpoint_path = CHECKPOINT_DIR / f"cycle_{cycle:02d}.pt"
    log_path = LOG_DIR / f"cycle_{cycle:02d}_train_log.csv"

    if not state_path.exists():
        raise FileNotFoundError(
            f"{state_path} does not exist.\n"
            f"Run random_selection.py and update_random_state.py for Cycle {cycle} first."
        )

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    train_ds = ActiveLearningDataset(state["labeled_train"], DATA_DIR, get_train_transforms())
    val_ds = DefectDataset(SPLIT_JSON, "val", DATA_DIR, get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("\n" + "=" * 60)
    print(f"RANDOM BASELINE - CYCLE {cycle}")
    print("=" * 60)
    print(f"Training samples   : {len(train_ds)}")
    print(f"Validation samples : {len(val_ds)}")
    print(f"Pool remaining     : {len(state['unlabeled_pool'])}")
    print("=" * 60 + "\n")

    device = get_device()
    model = build_model(freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_rows = []
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device, is_training=True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer, device, is_training=False
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        log_rows.append({
            "strategy": "random",
            "cycle": cycle,
            "epoch": epoch,
            "num_train_samples": len(train_ds),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"  -> New best random Cycle {cycle} model saved "
                f"(val_loss={val_loss:.4f})"
            )

    fieldnames = [
        "strategy", "cycle", "epoch", "num_train_samples",
        "train_loss", "train_acc", "val_loss", "val_acc"
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print("\nTraining complete.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training log   : {log_path}")


if __name__ == "__main__":
    main()
