"""
selection.py

Runs ONE active-learning selection cycle.

For the current pool:
    1. Randomly sample 40 candidates.
    2. Run the current ResNet50 checkpoint.
    3. Compute:
         - normalized predictive entropy (uncertainty)
         - embedding diversity
         - novelty distance from the ORIGINAL Cycle-0 training data (drift)
    4. Min-max normalize diversity and drift.
    5. Combine the three scores.
    6. Rank the 40 and select the top 20.

Outputs:
    selections/cycle_XX_candidates.csv
    selections/cycle_XX_to_label.csv
    selections/cycle_XX_embeddings.npz

The rejected 20 are NOT removed from the pool. Only active_learning.py changes state
after labels are supplied.
"""

from pathlib import Path
import argparse
import csv
import json
import math
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from dataset import ActiveLearningDataset, UnlabeledDataset

try:
    from transforms import get_eval_transforms
except ImportError as e:
    raise ImportError(
        "Could not import get_eval_transforms() from src/transforms.py."
    ) from e


PIPELINE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PIPELINE_DIR.parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
for _p in (MODEL_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA_DIR = PROJECT_ROOT / "raw_data"
STATES_DIR = PROJECT_ROOT / "states" / "resnet" / "old_pipeline" / "active"
SELECTIONS_DIR = PROJECT_ROOT / "selections" / "resnet" / "old_pipeline" / "active"

BASE_SEED = 42
CANDIDATE_SIZE = 40
SELECT_SIZE = 20


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    lo = float(values.min())
    hi = float(values.max())

    if math.isclose(lo, hi):
        return np.zeros_like(values)

    return (values - lo) / (hi - lo)


def normalized_entropy(probabilities):
    """
    Binary normalized predictive entropy.
    Returns a number in [0, 1].
    """
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1)
    return entropy / np.log(p.shape[1])


def pairwise_euclidean(a, b):
    """
    Pairwise Euclidean distances.
    a: [N, D], b: [M, D] -> [N, M]
    """
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    dist2 = np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)
    return np.sqrt(dist2 + 1e-12)


def diversity_score(candidate_embeddings):
    """
    Per-candidate diversity = nearest-neighbour distance within the 40-candidate batch.
    A larger value means the candidate is less redundant.
    """
    distances = pairwise_euclidean(candidate_embeddings, candidate_embeddings)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def novelty_score(candidate_embeddings, reference_embeddings):
    """
    Per-candidate drift/novelty score = distance to the nearest ORIGINAL training embedding.
    A larger value means the candidate is farther from Cycle-0 training data.
    """
    distances = pairwise_euclidean(candidate_embeddings, reference_embeddings)
    return distances.min(axis=1)


def build_model_for_checkpoint(device):
    """
    Assumes the baseline is a standard torchvision ResNet50 with a 2-class final FC layer.
    """
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format.")

    # Handle DataParallel checkpoints.
    cleaned = {}
    for key, value in state.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def load_checkpoint(path, device):
    model = build_model_for_checkpoint(device)
    checkpoint = torch.load(path, map_location=device)
    state = extract_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def infer_with_embeddings(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    all_embeddings = []
    all_paths = []

    captured = {}

    def fc_pre_hook(module, inputs):
        captured["features"] = inputs[0].detach()

    handle = model.fc.register_forward_pre_hook(fc_pre_hook)

    try:
        for images, paths in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_embeddings.append(captured["features"].cpu().numpy())
            all_paths.extend(list(paths))
    finally:
        handle.remove()

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_embeddings, axis=0),
        all_paths,
    )


@torch.no_grad()
def infer_training_embeddings(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embeddings = []
    captured = {}

    def fc_pre_hook(module, inputs):
        captured["features"] = inputs[0].detach()

    handle = model.fc.register_forward_pre_hook(fc_pre_hook)

    try:
        for images, _labels in loader:
            images = images.to(device)
            _ = model(images)
            all_embeddings.append(captured["features"].cpu().numpy())
    finally:
        handle.remove()

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True, help="Cycle to select, e.g. 1")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint from previous cycle")
    parser.add_argument(
        "--state",
        default=None,
        help="Current state JSON. Default: states/cycle_{cycle-1:02d}.json",
    )
    parser.add_argument(
        "--reference-state",
        default=None,
        help="Reference state for drift. Default: states/cycle_00.json",
    )
    parser.add_argument("--candidate-size", type=int, default=CANDIDATE_SIZE)
    parser.add_argument("--select-size", type=int, default=SELECT_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--w-uncertainty", type=float, default=1/3)
    parser.add_argument("--w-diversity", type=float, default=1/3)
    parser.add_argument("--w-drift", type=float, default=1/3)
    args = parser.parse_args()

    if args.cycle < 1:
        raise ValueError("Active-learning selection cycle must be >= 1.")

    state_path = (
        Path(args.state)
        if args.state
        else STATES_DIR / f"cycle_{args.cycle - 1:02d}.json"
    )
    reference_state_path = (
        Path(args.reference_state)
        if args.reference_state
        else STATES_DIR / "cycle_00.json"
    )
    checkpoint_path = Path(args.checkpoint)

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    with open(reference_state_path, "r", encoding="utf-8") as f:
        reference_state = json.load(f)

    current_pool = list(state["unlabeled_pool"])

    if len(current_pool) < args.candidate_size:
        raise ValueError(
            f"Pool contains only {len(current_pool)} images; "
            f"cannot sample {args.candidate_size}."
        )

    if args.select_size > args.candidate_size:
        raise ValueError("select-size cannot exceed candidate-size.")

    weights = np.array(
        [args.w_uncertainty, args.w_diversity, args.w_drift],
        dtype=np.float64,
    )
    if np.any(weights < 0) or math.isclose(float(weights.sum()), 0.0):
        raise ValueError("Weights must be non-negative and must not sum to zero.")
    weights = weights / weights.sum()

    # Reproducible but different random batch each cycle.
    rng = random.Random(BASE_SEED + args.cycle)
    candidates = rng.sample(current_pool, args.candidate_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint_path, device)
    eval_transform = get_eval_transforms()

    candidate_ds = UnlabeledDataset(
        candidates,
        DATA_DIR,
        transform=eval_transform,
    )

    reference_ds = ActiveLearningDataset(
        reference_state["labeled_train"],
        DATA_DIR,
        transform=eval_transform,
    )

    probabilities, candidate_embeddings, candidate_paths = infer_with_embeddings(
        model,
        candidate_ds,
        device,
        batch_size=args.batch_size,
    )

    reference_embeddings = infer_training_embeddings(
        model,
        reference_ds,
        device,
        batch_size=args.batch_size,
    )

    uncertainty = normalized_entropy(probabilities)

    diversity_raw = diversity_score(candidate_embeddings)
    drift_raw = novelty_score(candidate_embeddings, reference_embeddings)

    diversity_norm = minmax(diversity_raw)
    drift_norm = minmax(drift_raw)

    # Entropy is already normalized to [0, 1].
    combined = (
        weights[0] * uncertainty
        + weights[1] * diversity_norm
        + weights[2] * drift_norm
    )

    order = np.argsort(-combined)
    selected_indices = set(order[: args.select_size].tolist())

    rows = []
    for i, path in enumerate(candidate_paths):
        rows.append({
            "filename": path,
            "p_no_defect": float(probabilities[i, 0]),
            "p_defect": float(probabilities[i, 1]),
            "predicted_label": int(np.argmax(probabilities[i])),
            "uncertainty_entropy": float(uncertainty[i]),
            "diversity_raw": float(diversity_raw[i]),
            "diversity_norm": float(diversity_norm[i]),
            "drift_novelty_raw": float(drift_raw[i]),
            "drift_norm": float(drift_norm[i]),
            "combined_score": float(combined[i]),
            "selected": i in selected_indices,
        })

    rows.sort(key=lambda r: r["combined_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    candidate_csv = SELECTIONS_DIR / f"cycle_{args.cycle:02d}_candidates.csv"
    label_csv = SELECTIONS_DIR / f"cycle_{args.cycle:02d}_to_label.csv"
    embedding_npz = SELECTIONS_DIR / f"cycle_{args.cycle:02d}_embeddings.npz"

    fieldnames = [
        "rank",
        "filename",
        "p_no_defect",
        "p_defect",
        "predicted_label",
        "uncertainty_entropy",
        "diversity_raw",
        "diversity_norm",
        "drift_novelty_raw",
        "drift_norm",
        "combined_score",
        "selected",
    ]

    with open(candidate_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected_rows = [r for r in rows if r["selected"]]
    with open(label_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({"filename": row["filename"], "label": ""})

    # Save embeddings for KPI calculation.
    path_to_index = {p: i for i, p in enumerate(candidate_paths)}
    selected_mask = np.array(
        [p in {r["filename"] for r in selected_rows} for p in candidate_paths],
        dtype=bool,
    )

    np.savez_compressed(
        embedding_npz,
        candidate_embeddings=candidate_embeddings,
        reference_embeddings=reference_embeddings,
        candidate_paths=np.asarray(candidate_paths),
        selected_mask=selected_mask,
        probabilities=probabilities,
    )

    print(f"\nCycle {args.cycle} selection complete.")
    print(f"Current pool      : {len(current_pool)}")
    print(f"Random candidates : {len(candidates)}")
    print(f"Selected          : {len(selected_rows)}")
    print(f"Rejected/returned : {len(candidates) - len(selected_rows)}")
    print(f"\nCandidate scores: {candidate_csv}")
    print(f"Label these 20  : {label_csv}")
    print(f"Embeddings/KPIs : {embedding_npz}")

    print("\nTop 20:")
    for row in selected_rows:
        print(
            f"  #{row['rank']:02d} {row['filename']} "
            f"score={row['combined_score']:.4f}"
        )


if __name__ == "__main__":
    main()