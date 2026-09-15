"""
evaluate.py

Generic evaluation script for:
    - baseline model
    - active-learning cycle models
    - random-baseline cycle models

It evaluates ONE checkpoint on ONE fixed labeled split:
    val
    test
    shadow

Examples:

Baseline on test:
    python src/evaluate.py \
        --checkpoint checkpoints/resnet50_baseline_best.pt \
        --split test

Final active-learning model on test:
    python src/evaluate.py \
        --checkpoint checkpoints/active/cycle_05.pt \
        --split test

Final random model on shadow:
    python src/evaluate.py \
        --checkpoint checkpoints/random/cycle_05.pt \
        --split shadow
"""

from pathlib import Path
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from dataset import DefectDataset
from transforms import get_eval_transforms
from model import build_model


MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

SPLIT_JSON = PROJECT_ROOT / "split.json"
DATA_DIR = PROJECT_ROOT / "raw_data"

BATCH_SIZE = 16
NUM_WORKERS = 2


def get_device():
    if torch.cuda.is_available():
        print("GPU detected -- evaluation will use CUDA.")
        return torch.device("cuda")

    print("No GPU detected -- evaluation will use CPU.")
    return torch.device("cpu")


def extract_state_dict(checkpoint):
    """
    Accept both:
      torch.save(model.state_dict(), ...)
    and common wrapped checkpoint dictionaries.
    """

    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle checkpoints saved from DataParallel.
    cleaned = {}

    for name, tensor in state_dict.items():
        if name.startswith("module."):
            name = name[7:]

        cleaned[name] = tensor

    return cleaned


def load_model(checkpoint_path, device):
    """
    Rebuild the same 2-class model, then load saved weights.
    """

    model = build_model(
        freeze_backbone=True
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict(model, dataloader, device):
    y_true = []
    y_pred = []

    for images, labels in dataloader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.numpy().tolist())

    return (
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
    )


def calculate_metrics(y_true, y_pred):
    """
    Class mapping:
        0 = no_defect
        1 = defect
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    # Rows = actual, columns = predicted:
    #
    #                 predicted
    #                0        1
    # actual 0       TN       FP
    #        1       FN       TP
    #
    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------
    # Overall metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    # --------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------

    precision_per_class = precision_score(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    recall_per_class = recall_score(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    f1_per_class = f1_score(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    no_defect_precision = precision_per_class[0]
    defect_precision = precision_per_class[1]

    no_defect_recall = recall_per_class[0]
    defect_recall = recall_per_class[1]

    no_defect_f1 = f1_per_class[0]
    defect_f1 = f1_per_class[1]

    # --------------------------------------------------
    # Macro metrics
    # Equal importance given to both classes
    # --------------------------------------------------

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------
    # Weighted metrics
    # Takes class frequency into account
    # --------------------------------------------------

    weighted_precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------
    # False-negative rate for defect class
    # --------------------------------------------------

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        # Overall
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),

        # no_defect class
        "no_defect_precision": float(no_defect_precision),
        "no_defect_recall": float(no_defect_recall),
        "no_defect_f1": float(no_defect_f1),

        # defect class
        "defect_precision": float(defect_precision),
        "defect_recall": float(defect_recall),
        "defect_f1": float(defect_f1),
        "false_negative_rate": float(false_negative_rate),

        # Macro average
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),

        # Weighted average
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),

        # Confusion matrix
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the .pt checkpoint to evaluate.",
    )

    parser.add_argument(
        "--pipeline",
        choices=["old_pipeline", "new_pipeline"],
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=[
            "val",
            "test",
            "shadow",
        ],
        required=True,
        help="Which fixed labeled split to evaluate.",
    )

    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Optional result name, e.g. active_cycle_05. "
            "If omitted, the checkpoint filename is used."
        ),
    )

    args = parser.parse_args()

    RESULTS_DIR = PROJECT_ROOT / "results" / "resnet" / args.pipeline

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    dataset = DefectDataset(
        SPLIT_JSON,
        args.split,
        DATA_DIR,
        get_eval_transforms(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = get_device()

    model = load_model(
        checkpoint_path,
        device,
    )

    y_true, y_pred = predict(
        model,
        dataloader,
        device,
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print("\n" + "=" * 60)
    print("\nOverall metrics")
    print("-" * 40)
    print(f"Accuracy             : {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy    : {metrics['balanced_accuracy']:.4f}")

    print("\nPer-class metrics")
    print("-" * 40)

    print("NO_DEFECT (class 0)")
    print(f"Precision            : {metrics['no_defect_precision']:.4f}")
    print(f"Recall               : {metrics['no_defect_recall']:.4f}")
    print(f"F1                   : {metrics['no_defect_f1']:.4f}")

    print("\nDEFECT (class 1)")
    print(f"Precision            : {metrics['defect_precision']:.4f}")
    print(f"Recall               : {metrics['defect_recall']:.4f}")
    print(f"F1                   : {metrics['defect_f1']:.4f}")
    print(f"False-negative rate  : {metrics['false_negative_rate']:.4f}")

    print("\nMacro averages")
    print("-" * 40)
    print(f"Macro precision      : {metrics['macro_precision']:.4f}")
    print(f"Macro recall         : {metrics['macro_recall']:.4f}")
    print(f"Macro F1             : {metrics['macro_f1']:.4f}")


    print("\nWeighted averages")
    print("-" * 40)
    print(f"Weighted precision   : {metrics['weighted_precision']:.4f}")
    print(f"Weighted recall      : {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1          : {metrics['weighted_f1']:.4f}")

    cm = metrics["confusion_matrix"]

    print("\nConfusion matrix")
    print("-" * 40)
    print("                 Pred no_defect   Pred defect")
    print(
        f"Actual no_defect      {cm['tn']:4d}          {cm['fp']:4d}"
    )
    print(
        f"Actual defect         {cm['fn']:4d}          {cm['tp']:4d}"
    )

    # Save each evaluation so comparisons are easy later.
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_name = (
        args.name
        if args.name
        else checkpoint_path.stem
    )

    output_path = (
        RESULTS_DIR
        / f"{result_name}_{args.split}.json"
    )

    output = {
        "name": result_name,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_samples": len(dataset),
        **metrics,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()