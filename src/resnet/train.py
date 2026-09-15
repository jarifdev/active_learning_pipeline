"""
train.py

Trains the ResNet50 model for an active-learning cycle.

Cycle 0 = already-trained baseline.

Active-learning cycles use:
    states/cycle_01.json
    states/cycle_02.json
    ...
    states/cycle_05.json

Each state contains the growing labeled training set.

Examples:

    python src/train.py --cycle 1
    python src/train.py --cycle 2
"""

import time
import csv
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DefectDataset, ActiveLearningDataset
from transforms import get_train_transforms, get_eval_transforms
from model import build_model


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

SPLIT_JSON = PROJECT_ROOT / "split.json"
DATA_DIR = PROJECT_ROOT / "raw_data"



# ============================================================
# TRAINING SETTINGS
#
# Keep these the SAME as your baseline for a fair comparison.
# ============================================================

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 2

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        print("GPU detected -- training will use CUDA.")

        return torch.device("cuda")

    else:

        print(
            "No GPU detected -- "
            "training will use CPU (this will be slower)."
        )

        return torch.device("cpu")


# ============================================================
# RUN ONE EPOCH
# ============================================================

def run_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    is_training
):

    if is_training:
        model.train()

    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = (
        torch.enable_grad()
        if is_training
        else torch.no_grad()
    )

    with context:

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if is_training:

                loss.backward()

                optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cycle",
        type=int,
        required=True,
        help="Active-learning cycle number: 1 to 5"
    )

    parser.add_argument(
        "--pipeline",
        choices=["old_pipeline", "new_pipeline"],
        required=True,
        help="Experiment branch to read/write."
    )

    args = parser.parse_args()

    cycle = args.cycle
    pipeline = args.pipeline

    STATES_DIR = PROJECT_ROOT / "states" / "resnet" / pipeline / "active"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "resnet" / pipeline / "active"
    LOG_DIR = PROJECT_ROOT / "logs" / "resnet" / pipeline / "active"


    # --------------------------------------------------------
    # Validate cycle
    # --------------------------------------------------------

    if cycle < 1 or cycle > 5:

        raise ValueError(
            "Cycle must be between 1 and 5."
        )


    # --------------------------------------------------------
    # Set reproducible randomness
    # --------------------------------------------------------

    set_seed()


    # --------------------------------------------------------
    # Work out paths automatically
    # --------------------------------------------------------

    state_path = (
        STATES_DIR
        / f"cycle_{cycle:02d}.json"
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"cycle_{cycle:02d}.pt"
    )

    log_path = (
        LOG_DIR
        / f"cycle_{cycle:02d}_train_log.csv"
    )


    if not state_path.exists():

        raise FileNotFoundError(
            f"\nState file not found:\n"
            f"{state_path}\n\n"
            f"Complete the selection + labeling "
            f"step for Cycle {cycle} first."
        )


    # --------------------------------------------------------
    # Load active-learning state
    # --------------------------------------------------------

    with open(
        state_path,
        "r"
    ) as f:

        state = json.load(f)


    labeled_train = state["labeled_train"]


    # --------------------------------------------------------
    # Build training dataset
    #
    # IMPORTANT:
    # Training now comes from cycle_XX.json.
    # --------------------------------------------------------

    train_ds = ActiveLearningDataset(
        labeled_train,
        DATA_DIR,
        get_train_transforms()
    )


    # --------------------------------------------------------
    # Validation remains FIXED
    #
    # Always use the original val split.
    # --------------------------------------------------------

    val_ds = DefectDataset(
        SPLIT_JSON,
        "val",
        DATA_DIR,
        get_eval_transforms()
    )


    # --------------------------------------------------------
    # Dataloaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    print("\n" + "=" * 60)

    print(
        f"ACTIVE LEARNING CYCLE {cycle}"
    )

    print("=" * 60)

    print(
        f"Training samples   : {len(train_ds)}"
    )

    print(
        f"Validation samples : {len(val_ds)}"
    )

    print(
        f"Pool remaining     : "
        f"{len(state['unlabeled_pool'])}"
    )

    print("=" * 60 + "\n")


    # --------------------------------------------------------
    # Build a FRESH model
    #
    # Same training setup as baseline.
    # --------------------------------------------------------

    device = get_device()

    model = build_model(
        freeze_backbone=True
    )

    model = model.to(device)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Optimizer
    #
    # Only train parameters that require gradients.
    # --------------------------------------------------------

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Training variables
    # --------------------------------------------------------

    log_rows = []

    best_val_loss = float("inf")


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        start_time = time.time()


        # ----------------------
        # Training
        # ----------------------

        train_loss, train_acc = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            is_training=True
        )


        # ----------------------
        # Validation
        # ----------------------

        val_loss, val_acc = run_one_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            is_training=False
        )


        elapsed = (
            time.time()
            - start_time
        )


        print(

            f"Epoch "
            f"{epoch:2d}/{NUM_EPOCHS} | "

            f"train_loss="
            f"{train_loss:.4f} "

            f"train_acc="
            f"{train_acc:.4f} | "

            f"val_loss="
            f"{val_loss:.4f} "

            f"val_acc="
            f"{val_acc:.4f} | "

            f"{elapsed:.1f}s"
        )


        # ----------------------------------------------------
        # Save log row
        # ----------------------------------------------------

        log_rows.append({

            "cycle": cycle,

            "epoch": epoch,

            "num_train_samples":
                len(train_ds),

            "train_loss":
                train_loss,

            "train_acc":
                train_acc,

            "val_loss":
                val_loss,

            "val_acc":
                val_acc,
        })


        # ----------------------------------------------------
        # Save BEST model only
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"  -> New best Cycle "
                f"{cycle} model saved "
                f"(val_loss="
                f"{val_loss:.4f})"
            )


    # ========================================================
    # SAVE LOG
    # ========================================================

    fieldnames = [

        "cycle",

        "epoch",

        "num_train_samples",

        "train_loss",

        "train_acc",

        "val_loss",

        "val_acc"
    ]


    with open(
        log_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            log_rows
        )


    print("\nTraining complete.")

    print(
        f"Best checkpoint:\n"
        f"{checkpoint_path}"
    )

    print(
        f"\nTraining log:\n"
        f"{log_path}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()