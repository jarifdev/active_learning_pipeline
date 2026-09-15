"""
train_vit_ft2.py

Partial-fine-tuning version of the ACTIVE-LEARNING ViT trainer.

This deliberately reuses your EXISTING states/vit/active/cycle_XX.json.
Therefore it is a CONTROLLED TRAINING experiment:
    same AL-selected images,
    different ViT fine-tuning strategy.

It does NOT rerun acquisition/selection.

Outputs go to:
    checkpoints/vit_ft2/active/
    logs/vit_ft2/active/
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
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--pipeline", choices=["old_pipeline", "new_pipeline"], required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Use exactly the same epoch budget as baseline and random.",
    )
    args = parser.parse_args()

    STATES_DIR = PROJECT_ROOT / "states" / "vit" / args.pipeline / "active"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "vit" / args.pipeline / "active"
    LOG_DIR = PROJECT_ROOT / "logs" / "vit" / args.pipeline / "active"

    if not 1 <= args.cycle <= 5:
        raise ValueError("--cycle must be between 1 and 5")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    set_seed()

    state_path = STATES_DIR / f"cycle_{args.cycle:02d}.json"
    checkpoint_path = CHECKPOINT_DIR / f"cycle_{args.cycle:02d}.pt"
    log_path = LOG_DIR / f"cycle_{args.cycle:02d}_train_log.csv"

    if not state_path.exists():
        raise FileNotFoundError(f"Missing state: {state_path}")

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    train_ds = ActiveLearningDataset(
        state["labeled_train"], DATA_DIR, get_train_transforms()
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

    # Fresh ImageNet-pretrained ViT each cycle, as in the original experiment,
    # but now the final two transformer blocks are also allowed to adapt.
    model = build_vit(
        freeze_backbone=True,
        pretrained=True,
        train_last_n_blocks=TRAIN_LAST_N_BLOCKS,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    print("\n" + "=" * 65)
    print(f"VIT PARTIAL FINE-TUNING - ACTIVE CYCLE {args.cycle}")
    print("=" * 65)
    print(f"Train samples       : {len(train_ds)}")
    print(f"Validation samples  : {len(val_ds)}")
    print(f"Pool remaining      : {len(state['unlabeled_pool'])}")
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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

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
            "cycle": args.cycle,
            "epoch": epoch,
            "num_train_samples": len(train_ds),
            "pool_remaining": len(state["unlabeled_pool"]),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved new best checkpoint (val_loss={val_loss:.4f})")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Training log   : {log_path}")


if __name__ == "__main__":
    main()
