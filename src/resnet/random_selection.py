"""
random_selection.py

Runs ONE selection cycle for the RANDOM baseline.

For the current random-branch unlabeled pool:
    1. Randomly sample 40 candidates.
    2. Randomly select 20 of those 40 for labeling.
    3. Save all 40 candidates and the 20 selected images.

IMPORTANT:
- The other 20 are NOT removed from the pool here.
- update_random_state.py removes only the 20 selected/labeled images.
- Therefore the rejected 20 remain available in later cycles.

Creates:
    selections/random/cycle_XX_candidates.csv
    selections/random/cycle_XX_to_label.csv

Example:
    python src/random_selection.py --cycle 1
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
    parser.add_argument("--cycle", type=int, required=True, help="Random baseline cycle number: 1 to 5")
    parser.add_argument("--pipeline", choices=["old_pipeline", "new_pipeline"], required=True)
    parser.add_argument("--candidate-size", type=int, default=CANDIDATE_SIZE)
    parser.add_argument("--select-size", type=int, default=SELECT_SIZE)
    args = parser.parse_args()

    cycle = args.cycle
    STATES_DIR = PROJECT_ROOT / "states" / "resnet" / args.pipeline / "random"
    SELECTIONS_DIR = PROJECT_ROOT / "selections" / "resnet" / args.pipeline / "random"
    if cycle < 1 or cycle > 5:
        raise ValueError("Cycle must be between 1 and 5.")
    if args.select_size > args.candidate_size:
        raise ValueError("select-size cannot be greater than candidate-size.")

    state_path = STATES_DIR / f"cycle_{cycle - 1:02d}.json"
    if not state_path.exists():
        raise FileNotFoundError(f"{state_path} does not exist.\nComplete Cycle {cycle - 1} first.")

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    current_pool = list(state["unlabeled_pool"])
    if len(current_pool) < args.candidate_size:
        raise ValueError(f"Pool has only {len(current_pool)} images, but candidate-size={args.candidate_size}.")

    rng = random.Random(BASE_SEED + cycle)
    candidate_40 = rng.sample(current_pool, args.candidate_size)
    selected_20 = rng.sample(candidate_40, args.select_size)
    selected_set = set(selected_20)

    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    candidates_csv = SELECTIONS_DIR / f"cycle_{cycle:02d}_candidates.csv"
    labels_csv = SELECTIONS_DIR / f"cycle_{cycle:02d}_to_label.csv"

    with open(candidates_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "selected"])
        writer.writeheader()
        for filename in candidate_40:
            writer.writerow({"filename": filename, "selected": filename in selected_set})

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        for filename in selected_20:
            writer.writerow({"filename": filename, "label": ""})

    print(f"\nRANDOM BASELINE - CYCLE {cycle}")
    print("=" * 50)
    print(f"Current pool      : {len(current_pool)}")
    print(f"Random candidates : {len(candidate_40)}")
    print(f"Randomly selected : {len(selected_20)}")
    print(f"Returned to pool  : {len(candidate_40) - len(selected_20)}")
    print(f"\nAll 40 candidates:\n{candidates_csv}")
    print(f"\nFill labels for these 20:\n{labels_csv}")
    print(f"\nThen run: python src/update_random_state.py --cycle {cycle}")


if __name__ == "__main__":
    main()
