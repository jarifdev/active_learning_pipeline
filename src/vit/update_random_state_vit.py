"""
update_random_state_vit.py

Applies labels for one ViT RANDOM-control cycle.

Only the selected 20 are removed from the pool and added to training.
The rejected 20 remain in the pool.

Reads:
    states/vit/random/cycle_{cycle-1}.json
    selections/vit/random/cycle_XX_candidates.csv
    selections/vit/random/cycle_XX_to_label.csv

Creates:
    states/vit/random/cycle_XX.json
"""

from pathlib import Path
import argparse
import csv
import json

MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

LABEL_MAP = {
    "0": 0,
    "1": 1,
    "no_defect": 0,
    "defect": 1,
}


def read_labels(path):
    labels = {}

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename = row["filename"].strip()
            raw_label = row["label"].strip().lower()

            if not raw_label:
                raise ValueError(
                    f"Missing label for {filename}."
                )

            if raw_label not in LABEL_MAP:
                raise ValueError(
                    f"Invalid label '{row['label']}' for {filename}."
                )

            labels[filename] = LABEL_MAP[raw_label]

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["old_pipeline", "new_pipeline"], required=True)
    parser.add_argument(
        "--cycle",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    STATES_DIR = PROJECT_ROOT / "states" / "vit" / args.pipeline / "random"
    SELECTIONS_DIR = PROJECT_ROOT / "selections" / "vit" / args.pipeline / "random"

    cycle = args.cycle

    if cycle < 1 or cycle > 5:
        raise ValueError(
            "Cycle must be between 1 and 5."
        )

    previous_state_path = (
        STATES_DIR
        / f"cycle_{cycle - 1:02d}.json"
    )

    candidates_csv = (
        SELECTIONS_DIR
        / f"cycle_{cycle:02d}_candidates.csv"
    )

    labels_csv = (
        SELECTIONS_DIR
        / f"cycle_{cycle:02d}_to_label.csv"
    )

    output_state_path = (
        STATES_DIR
        / f"cycle_{cycle:02d}.json"
    )

    if output_state_path.exists():
        raise FileExistsError(
            f"{output_state_path} already exists."
        )

    with open(
        previous_state_path,
        "r",
        encoding="utf-8",
    ) as f:
        state = json.load(f)

    selected_paths = []

    with open(
        candidates_csv,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["selected"].strip().lower() == "true":
                selected_paths.append(
                    row["filename"]
                )

    labels = read_labels(labels_csv)

    if set(labels) != set(selected_paths):
        raise ValueError(
            "Label CSV must contain exactly the randomly selected images."
        )

    labeled_train = list(state["labeled_train"])
    pool = list(state["unlabeled_pool"])
    history = list(
        state.get("selected_history", [])
    )

    for path in selected_paths:
        if path not in pool:
            raise ValueError(
                f"{path} is not in current pool."
            )

        labeled_train.append({
            "path": path,
            "label": labels[path],
            "added_cycle": cycle,
        })

        pool.remove(path)

        history.append({
            "cycle": cycle,
            "path": path,
            "label": labels[path],
            "strategy": "random",
            "model": "vit_b_16",
        })

    next_state = {
        "cycle": cycle,
        "labeled_train": labeled_train,
        "unlabeled_pool": pool,
        "selected_history": history,
        "strategy": "random",
        "model": "vit_b_16",
    }

    STATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_state_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            next_state,
            f,
            indent=2,
        )

    print(f"Created: {output_state_path}")
    print(f"Training images: {len(labeled_train)}")
    print(f"Pool remaining : {len(pool)}")
    print("Rejected candidates remain in the pool.")


if __name__ == "__main__":
    main()
