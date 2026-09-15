"""
active_learning_vit.py

Applies human labels to the 20 images selected by selection_vit.py.

Reads:
    states/vit/active/cycle_{cycle-1}.json
    selections/vit/active/cycle_XX_candidates.csv
    selections/vit/active/cycle_XX_to_label.csv

Creates:
    states/vit/active/cycle_XX.json

Only selected images leave the pool. Rejected candidates remain.
"""

from pathlib import Path
import argparse
import csv
import json

PIPELINE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PIPELINE_DIR.parent
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

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not {"filename", "label"}.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                f"{path} must contain filename,label columns."
            )

        for row in reader:
            filename = row["filename"].strip()
            raw_label = row["label"].strip().lower()

            if not raw_label:
                raise ValueError(
                    f"Missing label for {filename}."
                )

            if raw_label not in LABEL_MAP:
                raise ValueError(
                    f"Invalid label '{row['label']}' for {filename}. "
                    "Use 0/1 or no_defect/defect."
                )

            if filename in labels:
                raise ValueError(
                    f"Duplicate filename: {filename}"
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

    STATES_DIR = PROJECT_ROOT / "states" / "vit" / args.pipeline / "active"
    SELECTIONS_DIR = PROJECT_ROOT / "selections" / "vit" / args.pipeline / "active"

    cycle = args.cycle

    if cycle < 1 or cycle > 5:
        raise ValueError(
            "Cycle must be between 1 and 5."
        )

    previous_state_path = (
        STATES_DIR
        / f"cycle_{cycle - 1:02d}.json"
    )

    candidate_csv = (
        SELECTIONS_DIR
        / f"cycle_{cycle:02d}_candidates.csv"
    )

    label_csv = (
        SELECTIONS_DIR
        / f"cycle_{cycle:02d}_to_label.csv"
    )

    output_path = (
        STATES_DIR
        / f"cycle_{cycle:02d}.json"
    )

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists."
        )

    with open(
        previous_state_path,
        "r",
        encoding="utf-8",
    ) as f:
        state = json.load(f)

    selected_paths = []

    with open(
        candidate_csv,
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

    labels = read_labels(label_csv)

    if set(labels) != set(selected_paths):
        missing = set(selected_paths) - set(labels)
        extra = set(labels) - set(selected_paths)

        raise ValueError(
            "Label file must contain exactly the selected images.\n"
            f"Missing={sorted(missing)}\n"
            f"Extra={sorted(extra)}"
        )

    pool = list(state["unlabeled_pool"])
    labeled_train = list(state["labeled_train"])
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
            "strategy": "active",
            "model": "vit_b_16",
        })

    next_state = {
        "cycle": cycle,
        "labeled_train": labeled_train,
        "unlabeled_pool": pool,
        "selected_history": history,
        "strategy": "active",
        "model": "vit_b_16",
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            next_state,
            f,
            indent=2,
        )

    print(f"Created {output_path}")
    print(f"  cycle          : {cycle}")
    print(f"  labeled_train  : {len(labeled_train)}")
    print(f"  unlabeled_pool : {len(pool)}")
    print(f"  added this cycle: {len(selected_paths)}")
    print("  rejected candidates remain in the pool.")


if __name__ == "__main__":
    main()
