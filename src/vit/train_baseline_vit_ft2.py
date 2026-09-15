"""
train_baseline_vit_ft2.py

Controlled partial-fine-tuning experiment for the ViT baseline.

IMPORTANT:
- Uses the SAME original 160-image training split.
- Unfreezes only the last 2 transformer blocks + final LayerNorm + head.
- Saves to checkpoints/vit_ft2/ so the completed frozen-ViT experiment
  is NOT overwritten.
- Choose the epoch budget using validation behavior, then use exactly
  the same --epochs value for baseline, AL, and random.
"""


from pathlib import Path
import argparse
import csv
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Scripts live in src/vit/, while dataset.py/transforms.py live in src/.
MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
for _p in (MODEL_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataset import DefectDataset, ActiveLearningDataset
from transforms import get_train_transforms, get_eval_transforms
from model_vit_ft2 import (
    build_vit,
    get_finetune_parameter_groups,
    count_parameters,
)

PROJECT_ROOT = SRC_DIR.parent
SPLIT_JSON = PROJECT_ROOT / "split.json"
DATA_DIR = PROJECT_ROOT / "raw_data"

BATCH_SIZE = 16
HEAD_LR = 1e-3
BACKBONE_LR = 1e-5
NUM_WORKERS = 2
SEED = 42
TRAIN_LAST_N_BLOCKS = 2


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


def run_one_epoch(model, loader, criterion, optimizer, device, is_training):
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, labels in loader:
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
    parser.add_argument("--pipeline", choices=["old_pipeline", "new_pipeline"], required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Fixed epoch budget. Use the same value for baseline, AL, and random.",
    )
    args = parser.parse_args()
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vit" / args.pipeline / "baseline" / "best.pt"
    LOG_PATH = PROJECT_ROOT / "logs" / "vit" / args.pipeline / "baseline" / "train_log.csv"

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    set_seed()

    train_ds = DefectDataset(
        SPLIT_JSON, "train", DATA_DIR, get_train_transforms()
    )
    val_ds = DefectDataset(
        SPLIT_JSON, "val", DATA_DIR, get_eval_transforms()
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    device = get_device()

    model = build_vit(
        freeze_backbone=True,
        pretrained=True,
        train_last_n_blocks=TRAIN_LAST_N_BLOCKS,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    print("\n" + "=" * 65)
    print("VIT PARTIAL FINE-TUNING - BASELINE")
    print("=" * 65)
    print(f"Train samples       : {len(train_ds)}")
    print(f"Validation samples  : {len(val_ds)}")
    print(f"Train last blocks   : {TRAIN_LAST_N_BLOCKS}")
    print(f"Head LR             : {HEAD_LR}")
    print(f"Backbone LR         : {BACKBONE_LR}")
    print(f"Epochs              : {args.epochs}")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    print("=" * 65 + "\n")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        get_finetune_parameter_groups(
            model,
            head_lr=HEAD_LR,
            backbone_lr=BACKBONE_LR,
        )
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    rows = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device, True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer, device, False
        )

        elapsed = time.time() - start

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        rows.append({
            "epoch": epoch,
            "num_train_samples": len(train_ds),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Saved new best checkpoint (val_loss={val_loss:.4f})")

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBest checkpoint: {CHECKPOINT_PATH}")
    print(f"Training log   : {LOG_PATH}")


if __name__ == "__main__":
    main()
