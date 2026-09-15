"""
update_random_state.py

Applies the labels from one RANDOM baseline cycle.

Reads:
    states/random/cycle_{cycle-1}.json
    selections/random/cycle_XX_candidates.csv
    selections/random/cycle_XX_to_label.csv

Creates:
    states/random/cycle_XX.json

Only the randomly selected 20 images are:
    - added to labeled_train
    - removed from unlabeled_pool

The other 20 candidates remain in the unlabeled pool.
"""

from pathlib import Path
import argparse
import csv
import json

MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

LABEL_MAP = {"0": 0, "1": 1, "no_defect": 0, "defect": 1}


def read_labels(path):
    labels = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"filename", "label"}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must contain filename,label columns.")
        for row in reader:
            filename = row["filename"].strip()
            raw_label = row["label"].strip().lower()
            if not raw_label:
                raise ValueError(f"Missing label for {filename}.")
            if raw_label not in LABEL_MAP:
                raise ValueError(f"Invalid label '{row['label']}' for {filename}. Use 0/1 or no_defect/defect.")
            if filename in labels:
                raise ValueError(f"Duplicate image in label CSV: {filename}")
            labels[filename] = LABEL_MAP[raw_label]
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True, help="Random baseline cycle number: 1 to 5")
    parser.add_argument("--pipeline", choices=["old_pipeline", "new_pipeline"], required=True)
    args = parser.parse_args()
    cycle = args.cycle
    STATES_DIR = PROJECT_ROOT / "states" / "resnet" / args.pipeline / "random"
    SELECTIONS_DIR = PROJECT_ROOT / "selections" / "resnet" / args.pipeline / "random"

    if cycle < 1 or cycle > 5:
        raise ValueError("Cycle must be between 1 and 5.")

    previous_state_path = STATES_DIR / f"cycle_{cycle - 1:02d}.json"
    candidates_csv = SELECTIONS_DIR / f"cycle_{cycle:02d}_candidates.csv"
    labels_csv = SELECTIONS_DIR / f"cycle_{cycle:02d}_to_label.csv"
    output_state_path = STATES_DIR / f"cycle_{cycle:02d}.json"

    if output_state_path.exists():
        raise FileExistsError(
            f"{output_state_path} already exists.\n"
            "Delete it manually only if you intentionally want to redo this random cycle."
        )

    with open(previous_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    selected_paths = []
    with open(candidates_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["selected"].strip().lower() == "true":
                selected_paths.append(row["filename"])

    labels = read_labels(labels_csv)
    if set(labels) != set(selected_paths):
        missing = set(selected_paths) - set(labels)
        extra = set(labels) - set(selected_paths)
        raise ValueError(
            "Label CSV must contain exactly the 20 selected images.\n"
            f"Missing: {sorted(missing)}\nExtra: {sorted(extra)}"
        )

    labeled_train = list(state["labeled_train"])
    pool = list(state["unlabeled_pool"])
    history = list(state.get("selected_history", []))

    for path in selected_paths:
        if path not in pool:
            raise ValueError(f"{path} is not in the current unlabeled pool.")

        labeled_train.append({"path": path, "label": labels[path], "added_cycle": cycle})
        pool.remove(path)
        history.append({"cycle": cycle, "path": path, "label": labels[path], "strategy": "random"})

    next_state = {
        "cycle": cycle,
        "labeled_train": labeled_train,
        "unlabeled_pool": pool,
        "selected_history": history,
        "strategy": "random",
    }

    STATES_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_state_path, "w", encoding="utf-8") as f:
        json.dump(next_state, f, indent=2)

    print(f"\nCreated: {output_state_path}")
    print(f"Cycle            : {cycle}")
    print(f"Training images  : {len(labeled_train)}")
    print(f"Pool remaining   : {len(pool)}")
    print(f"Added this cycle : {len(selected_paths)}")
    print("\nThe rejected 20 candidates remain in the pool.")


if __name__ == "__main__":
    main()
