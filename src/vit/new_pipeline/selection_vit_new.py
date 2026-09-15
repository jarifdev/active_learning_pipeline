"""
selection_vit_new.py

Improved active-learning selection for the ViT experiment.

Strategy per cycle:
    1. Run inference on the ENTIRE current unlabeled pool.
    2. Compute normalized predictive entropy for every pool image.
    3. Compute drift/novelty for every pool image relative to the ORIGINAL
       Cycle-0 labeled training set.
    4. Min-max normalize drift.
    5. Compute an information score:
           information_score =
               w_uncertainty * uncertainty
             + w_drift       * drift_norm
       Default weights are 0.5 / 0.5.
    6. Keep the top 40 images by information score.
    7. Apply k-center greedy to the ViT embeddings of those top 40.
    8. Select exactly 20 images for human labeling.

Important:
    - Drift reference images remain the original Cycle-0 labeled 160.
    - Diversity is NOT included in the weighted information score.
      Diversity is handled explicitly by k-center greedy.
    - The first k-center point is the highest-information candidate.
    - The remaining top-40 images that are not selected stay in the pool.

Outputs remain compatible with active_learning_vit.py:
    selections/vit/active/cycle_XX_candidates.csv
    selections/vit/active/cycle_XX_to_label.csv
    selections/vit/active/cycle_XX_embeddings.npz

Additional audit output:
    selections/vit/active/cycle_XX_pool_scores.csv

Example:
    python src/selection_vit_new.py \
        --cycle 1 \
        --checkpoint checkpoints/vit/baseline.pt
"""

from pathlib import Path
import argparse
import csv
import json
import math
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ActiveLearningDataset, UnlabeledDataset
from transforms import get_eval_transforms
from model_vit import build_vit, get_feature_layer


PIPELINE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PIPELINE_DIR.parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
for _p in (MODEL_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA_DIR = PROJECT_ROOT / "raw_data"
STATES_DIR = PROJECT_ROOT / "states" / "vit" / "new_pipeline" / "active"
SELECTIONS_DIR = PROJECT_ROOT / "selections" / "vit" / "new_pipeline" / "active"

CANDIDATE_SIZE = 40
SELECT_SIZE = 20


def minmax(values):
    """Min-max normalize a 1-D array to [0, 1]."""
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return values

    lo = float(values.min())
    hi = float(values.max())

    if math.isclose(lo, hi):
        return np.zeros_like(values)

    return (values - lo) / (hi - lo)


def normalized_entropy(probabilities):
    """
    Normalized predictive entropy.

    For binary classification:
        H(p) = -sum_c p_c log(p_c) / log(2)

    Result lies approximately in [0, 1].
    """
    p = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1e-12,
        1.0,
    )
    entropy = -np.sum(p * np.log(p), axis=1)
    return entropy / np.log(p.shape[1])


def pairwise_euclidean(a, b):
    """Efficient pairwise Euclidean distance matrix."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T

    dist2 = np.maximum(
        a2 + b2 - 2.0 * a @ b.T,
        0.0,
    )
    return np.sqrt(dist2 + 1e-12)


def novelty_score(pool_embeddings, reference_embeddings):
    """
    Drift / novelty score.

    For each pool image, use its Euclidean distance to the nearest embedding
    from the ORIGINAL Cycle-0 labeled training set.

    Higher = more novel / farther from the original labeled distribution.
    """
    distances = pairwise_euclidean(
        pool_embeddings,
        reference_embeddings,
    )
    return distances.min(axis=1)


def kcenter_greedy(embeddings, select_size, first_index=0):
    """
    K-center greedy batch selection.

    embeddings:
        Embeddings of the already preselected candidate set (e.g. top 40).

    first_index:
        Index of the first chosen candidate. We use index 0 because the
        candidate set is ordered by information score descending, so index 0
        is the highest-information candidate.

    At every subsequent step, choose the point whose distance to its nearest
    already-selected point is maximal.

    Returns:
        selected_indices: list of candidate-local indices in selection order
        final_min_distances: nearest-selected distance for every candidate
                             after the final batch has been selected
    """
    x = np.asarray(embeddings, dtype=np.float64)
    n = len(x)

    if n == 0:
        raise ValueError("Cannot run k-center on an empty candidate set.")

    if select_size < 1:
        raise ValueError("select_size must be at least 1.")

    if select_size > n:
        raise ValueError(
            f"select_size={select_size} exceeds candidate count={n}."
        )

    if not (0 <= first_index < n):
        raise ValueError("first_index is outside the candidate set.")

    selected = [int(first_index)]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[first_index] = True

    # Initial distance to the first selected center.
    min_distances = pairwise_euclidean(
        x,
        x[[first_index]],
    )[:, 0]
    min_distances[first_index] = 0.0

    while len(selected) < select_size:
        # Never select an already-selected point.
        scores = min_distances.copy()
        scores[selected_mask] = -np.inf

        next_index = int(np.argmax(scores))
        selected.append(next_index)
        selected_mask[next_index] = True

        # Update each point's distance to its nearest selected center.
        distance_to_new = pairwise_euclidean(
            x,
            x[[next_index]],
        )[:, 0]
        min_distances = np.minimum(
            min_distances,
            distance_to_new,
        )
        min_distances[selected_mask] = 0.0

    return selected, min_distances


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format.")

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value

    return cleaned


def load_checkpoint(path, device):
    """
    Rebuild ViT architecture without downloading pretrained weights and load
    all trained weights from the supplied checkpoint.
    """
    model = build_vit(
        freeze_backbone=True,
        pretrained=False,
    )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        extract_state_dict(checkpoint),
        strict=True,
    )

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_with_embeddings(model, dataset, device, batch_size=16):
    """
    Run inference and capture the representation entering the ViT classifier
    head. Returns probabilities, embeddings, and paths in dataset order.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_probs = []
    all_embeddings = []
    all_paths = []

    captured = {}

    def feature_pre_hook(module, inputs):
        captured["features"] = inputs[0].detach()

    handle = get_feature_layer(model).register_forward_pre_hook(
        feature_pre_hook
    )

    try:
        for images, paths in loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_embeddings.append(
                captured["features"].cpu().numpy()
            )
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
    """Extract embeddings for the fixed Cycle-0 labeled reference set."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_embeddings = []
    captured = {}

    def feature_pre_hook(module, inputs):
        captured["features"] = inputs[0].detach()

    handle = get_feature_layer(model).register_forward_pre_hook(
        feature_pre_hook
    )

    try:
        for images, _labels in loader:
            images = images.to(device)
            _ = model(images)

            all_embeddings.append(
                captured["features"].cpu().numpy()
            )
    finally:
        handle.remove()

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="ViT checkpoint from the previous cycle.",
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=CANDIDATE_SIZE,
        help="Top-N information-score images passed to k-center.",
    )
    parser.add_argument(
        "--select-size",
        type=int,
        default=SELECT_SIZE,
        help="Number of images finally selected for labeling.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--w-uncertainty",
        type=float,
        default=0.5,
        help="Weight of normalized entropy in preselection score.",
    )
    parser.add_argument(
        "--w-drift",
        type=float,
        default=0.5,
        help="Weight of normalized drift in preselection score.",
    )

    args = parser.parse_args()

    if args.cycle < 1 or args.cycle > 5:
        raise ValueError("Cycle must be between 1 and 5.")

    if args.candidate_size < 1:
        raise ValueError("candidate-size must be at least 1.")

    if args.select_size < 1:
        raise ValueError("select-size must be at least 1.")

    if args.select_size > args.candidate_size:
        raise ValueError(
            "select-size cannot exceed candidate-size."
        )

    weights = np.array(
        [args.w_uncertainty, args.w_drift],
        dtype=np.float64,
    )

    if np.any(weights < 0) or math.isclose(float(weights.sum()), 0.0):
        raise ValueError(
            "Uncertainty/drift weights must be non-negative "
            "and must not sum to zero."
        )

    # Normalize supplied weights, so e.g. 1/1 behaves like 0.5/0.5.
    weights = weights / weights.sum()

    current_state_path = (
        STATES_DIR
        / f"cycle_{args.cycle - 1:02d}.json"
    )

    # Drift reference IMAGE SET stays fixed to original Cycle-0 train.
    reference_state_path = STATES_DIR / "cycle_00.json"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not current_state_path.exists():
        raise FileNotFoundError(
            f"Missing state: {current_state_path}"
        )

    if not reference_state_path.exists():
        raise FileNotFoundError(
            f"Missing Cycle-0 state: {reference_state_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    with open(current_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    with open(reference_state_path, "r", encoding="utf-8") as f:
        reference_state = json.load(f)

    current_pool = list(state["unlabeled_pool"])

    if len(current_pool) < args.candidate_size:
        raise ValueError(
            f"Pool contains only {len(current_pool)} images; "
            f"cannot keep top {args.candidate_size}."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nViT Cycle {args.cycle} - improved active selection")
    print(f"Device                 : {device}")
    print(f"Current pool           : {len(current_pool)}")
    print(f"Scoring pool images    : {len(current_pool)}")
    print(f"Top candidates retained: {args.candidate_size}")
    print(f"Final label budget     : {args.select_size}")
    print(
        "Preselection weights   : "
        f"uncertainty={weights[0]:.3f}, drift={weights[1]:.3f}"
    )

    model = load_checkpoint(
        checkpoint_path,
        device,
    )

    eval_transform = get_eval_transforms()

    # NEW: infer on the WHOLE current unlabeled pool.
    pool_ds = UnlabeledDataset(
        current_pool,
        DATA_DIR,
        transform=eval_transform,
    )

    # Fixed reference image set: original Cycle-0 labeled train.
    reference_ds = ActiveLearningDataset(
        reference_state["labeled_train"],
        DATA_DIR,
        transform=eval_transform,
    )

    probabilities, pool_embeddings, pool_paths = infer_with_embeddings(
        model,
        pool_ds,
        device,
        batch_size=args.batch_size,
    )

    reference_embeddings = infer_training_embeddings(
        model,
        reference_ds,
        device,
        batch_size=args.batch_size,
    )

    # ----- Stage 1: information preselection -----
    uncertainty = normalized_entropy(probabilities)

    drift_raw = novelty_score(
        pool_embeddings,
        reference_embeddings,
    )
    drift_norm = minmax(drift_raw)

    information_score = (
        weights[0] * uncertainty
        + weights[1] * drift_norm
    )

    # Stable sort keeps deterministic ordering for exact ties.
    info_order = np.argsort(
        -information_score,
        kind="stable",
    )

    top_indices = info_order[: args.candidate_size]

    candidate_paths = [pool_paths[i] for i in top_indices]
    candidate_probabilities = probabilities[top_indices]
    candidate_embeddings = pool_embeddings[top_indices]
    candidate_uncertainty = uncertainty[top_indices]
    candidate_drift_raw = drift_raw[top_indices]
    candidate_drift_norm = drift_norm[top_indices]
    candidate_information = information_score[top_indices]

    # ----- Stage 2: explicit batch diversity with k-center -----
    # candidate index 0 is the highest-information image.
    selected_local_order, final_min_distances = kcenter_greedy(
        candidate_embeddings,
        select_size=args.select_size,
        first_index=0,
    )

    selected_local_set = set(selected_local_order)
    kcenter_position = {
        idx: pos + 1
        for pos, idx in enumerate(selected_local_order)
    }

    # Build top-40 candidate rows.
    candidate_rows = []

    for local_i, path in enumerate(candidate_paths):
        candidate_rows.append({
            "rank": local_i + 1,
            "filename": path,
            "p_no_defect": float(candidate_probabilities[local_i, 0]),
            "p_defect": float(candidate_probabilities[local_i, 1]),
            "predicted_label": int(
                np.argmax(candidate_probabilities[local_i])
            ),
            "uncertainty_entropy": float(
                candidate_uncertainty[local_i]
            ),
            "drift_novelty_raw": float(
                candidate_drift_raw[local_i]
            ),
            "drift_norm": float(
                candidate_drift_norm[local_i]
            ),
            "information_score": float(
                candidate_information[local_i]
            ),
            "selected": local_i in selected_local_set,
            "kcenter_order": (
                kcenter_position[local_i]
                if local_i in kcenter_position
                else ""
            ),
            # Useful audit statistic after the final k-center batch:
            # 0 for selected points; >0 for non-selected top-40 points.
            "distance_to_nearest_selected_final": float(
                final_min_distances[local_i]
            ),
        })

    # Full-pool audit rows: lets you prove every pool image was scored.
    top_index_set = set(top_indices.tolist())
    pool_rank = np.empty(len(pool_paths), dtype=int)
    pool_rank[info_order] = np.arange(1, len(pool_paths) + 1)

    pool_rows = []
    for i, path in enumerate(pool_paths):
        pool_rows.append({
            "rank": int(pool_rank[i]),
            "filename": path,
            "p_no_defect": float(probabilities[i, 0]),
            "p_defect": float(probabilities[i, 1]),
            "predicted_label": int(np.argmax(probabilities[i])),
            "uncertainty_entropy": float(uncertainty[i]),
            "drift_novelty_raw": float(drift_raw[i]),
            "drift_norm": float(drift_norm[i]),
            "information_score": float(information_score[i]),
            "preselected_top40": i in top_index_set,
        })

    pool_rows.sort(
        key=lambda row: row["rank"]
    )

    SELECTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pool_scores_csv = (
        SELECTIONS_DIR
        / f"cycle_{args.cycle:02d}_pool_scores.csv"
    )
    candidate_csv = (
        SELECTIONS_DIR
        / f"cycle_{args.cycle:02d}_candidates.csv"
    )
    label_csv = (
        SELECTIONS_DIR
        / f"cycle_{args.cycle:02d}_to_label.csv"
    )
    embedding_npz = (
        SELECTIONS_DIR
        / f"cycle_{args.cycle:02d}_embeddings.npz"
    )

    pool_fieldnames = [
        "rank",
        "filename",
        "p_no_defect",
        "p_defect",
        "predicted_label",
        "uncertainty_entropy",
        "drift_novelty_raw",
        "drift_norm",
        "information_score",
        "preselected_top40",
    ]

    with open(
        pool_scores_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=pool_fieldnames,
        )
        writer.writeheader()
        writer.writerows(pool_rows)

    candidate_fieldnames = [
        "rank",
        "filename",
        "p_no_defect",
        "p_defect",
        "predicted_label",
        "uncertainty_entropy",
        "drift_novelty_raw",
        "drift_norm",
        "information_score",
        "selected",
        "kcenter_order",
        "distance_to_nearest_selected_final",
    ]

    with open(
        candidate_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=candidate_fieldnames,
        )
        writer.writeheader()
        writer.writerows(candidate_rows)

    # Preserve actual k-center selection order in the to-label file.
    selected_rows_in_order = [
        candidate_rows[i]
        for i in selected_local_order
    ]

    with open(
        label_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "label"],
        )
        writer.writeheader()

        for row in selected_rows_in_order:
            writer.writerow({
                "filename": row["filename"],
                "label": "",
            })

    selected_mask = np.array(
        [
            i in selected_local_set
            for i in range(len(candidate_paths))
        ],
        dtype=bool,
    )

    np.savez_compressed(
        embedding_npz,
        candidate_embeddings=candidate_embeddings,
        reference_embeddings=reference_embeddings,
        candidate_paths=np.asarray(candidate_paths),
        selected_mask=selected_mask,
        probabilities=candidate_probabilities,
        uncertainty_entropy=candidate_uncertainty,
        drift_novelty_raw=candidate_drift_raw,
        drift_norm=candidate_drift_norm,
        information_score=candidate_information,
        kcenter_selection_order=np.asarray(
            selected_local_order,
            dtype=np.int64,
        ),
        final_nearest_selected_distance=final_min_distances,
    )

    print(f"\nCycle {args.cycle} selection complete.")
    print(f"Whole pool scored      : {len(current_pool)}")
    print(f"Top-info candidates    : {len(candidate_rows)}")
    print(f"K-center selected      : {len(selected_rows_in_order)}")
    print(
        f"Top-40 not labeled     : "
        f"{len(candidate_rows) - len(selected_rows_in_order)}"
    )
    print(
        f"Total pool not labeled : "
        f"{len(current_pool) - len(selected_rows_in_order)}"
    )
    print(f"\nAll-pool scores        : {pool_scores_csv}")
    print(f"Top-40 candidates      : {candidate_csv}")
    print(f"Label these 20         : {label_csv}")
    print(f"Embeddings/KPI data    : {embedding_npz}")


if __name__ == "__main__":
    main()
