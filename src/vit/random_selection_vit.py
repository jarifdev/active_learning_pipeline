"""
random_selection_vit.py

Runs one RANDOM-control selection cycle for the independent ViT replication.

Current pool -> random 40 -> random 20 to label.
The rejected 20 are not removed here and remain available later.

Creates:
    selections/vit/random/cycle_XX_candidates.csv
    selections/vit/random/cycle_XX_to_label.csv

Run:
    python src/random_selection_vit.py --cycle 1
"""

from pathlib import Path
import argparse
import csv
import json
import random

MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

BASE_SEED = 42
CANDIDATE_SIZE = 40
SELECT_SIZE = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["old_pipeline", "new_pipeline"], required=True)
    parser.add_argument(
        "--cycle",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=CANDIDATE_SIZE,
    )
    parser.add_argument(
        "--select-size",
        type=int,
        default=SELECT_SIZE,
    )
    args = parser.parse_args()

    STATES_DIR = PROJECT_ROOT / "states" / "vit" / args.pipeline / "random"
    SELECTIONS_DIR = PROJECT_ROOT / "selections" / "vit" / args.pipeline / "random"

    cycle = args.cycle

    if cycle < 1 or cycle > 5:
        raise ValueError(
            "Cycle must be between 1 and 5."
        )

    if args.select_size > args.candidate_size:
        raise ValueError(
            "select-size cannot exceed candidate-size."
        )

    state_path = (
        STATES_DIR
        / f"cycle_{cycle - 1:02d}.json"
    )

    if not state_path.exists():
        raise FileNotFoundError(
            f"{state_path} does not exist."
        )

    with open(
        state_path,
        "r",
        encoding="utf-8",
    ) as f:
        state = json.load(f)

    current_pool = list(state["unlabeled_pool"])

    if len(current_pool) < args.candidate_size:
        raise ValueError(
            f"Pool has only {len(current_pool)} images."
        )

    rng = random.Random(BASE_SEED + cycle)

    candidates = rng.sample(
        current_pool,
        args.candidate_size,
    )

    selected = rng.sample(
        candidates,
        args.select_size,
    )

    selected_set = set(selected)

    SELECTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates_csv = (
        SELECTIONS_DIR
        / f"cycle_{cycle:02d}_candidates.csv"
    )

    labels_csv = (
        SELECTIONS_DIR
        / f"cycle_{cycle:02d}_to_label.csv"
    )

    with open(
        candidates_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "selected"],
        )
        writer.writeheader()

        for filename in candidates:
            writer.writerow({
                "filename": filename,
                "selected": filename in selected_set,
            })

    with open(
        labels_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "label"],
        )
        writer.writeheader()

        for filename in selected:
            writer.writerow({
                "filename": filename,
                "label": "",
            })

    print(f"\nViT RANDOM - CYCLE {cycle}")
    print(f"Current pool      : {len(current_pool)}")
    print(f"Random candidates : {len(candidates)}")
    print(f"Randomly selected : {len(selected)}")
    print(
        f"Returned to pool  : "
        f"{len(candidates) - len(selected)}"
    )
    print(f"\nFill labels in: {labels_csv}")


if __name__ == "__main__":
    main()
