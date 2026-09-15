"""
active_learning.py

Applies human labels to the selected 20 images and creates the next state.

Workflow:
    selection.py -> cycle_XX_to_label.csv
    human fills in label column
    active_learning.py -> states/cycle_XX.json

Only selected/labeled images are removed from the unlabeled pool.
The rejected 20 remain in the pool automatically.
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

        required = {"filename", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain columns: filename,label"
            )

        for row in reader:
            filename = row["filename"].strip()
            raw_label = row["label"].strip().lower()

            if not raw_label:
                raise ValueError(f"Missing label for {filename}")

            if raw_label not in LABEL_MAP:
                raise ValueError(
                    f"Invalid label '{row['label']}' for {filename}. "
                    "Use 0/1 or no_defect/defect."
                )

            if filename in labels:
                raise ValueError(f"Duplicate filename in label file: {filename}")

            labels[filename] = LABEL_MAP[raw_label]

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["old_pipeline", "new_pipeline"], required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument(
        "--previous-state",
        default=None,
        help="Default: states/cycle_{cycle-1:02d}.json",
    )
    parser.add_argument(
        "--selection-csv",
        default=None,
        help="Default: selections/cycle_XX_candidates.csv",
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="Default: selections/cycle_XX_to_label.csv",
    )
    args = parser.parse_args()

    STATES_DIR = PROJECT_ROOT / "states" / "resnet" / args.pipeline / "active"
    SELECTIONS_DIR = PROJECT_ROOT / "selections" / "resnet" / args.pipeline / "active"

    if args.cycle < 1:
        raise ValueError("cycle must be >= 1")

    previous_state_path = (
        Path(args.previous_state)
        if args.previous_state
        else STATES_DIR / f"cycle_{args.cycle - 1:02d}.json"
    )
    selection_csv = (
        Path(args.selection_csv)
        if args.selection_csv
        else SELECTIONS_DIR / f"cycle_{args.cycle:02d}_candidates.csv"
    )
    labels_csv = (
        Path(args.labels_csv)
        if args.labels_csv
        else SELECTIONS_DIR / f"cycle_{args.cycle:02d}_to_label.csv"
    )

    with open(previous_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    selected_paths = []
    with open(selection_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["selected"].strip().lower() == "true":
                selected_paths.append(row["filename"])

    labels = read_labels(labels_csv)

    if set(labels) != set(selected_paths):
        missing = set(selected_paths) - set(labels)
        extra = set(labels) - set(selected_paths)
        raise ValueError(
            f"Label file must contain exactly the selected images. "
            f"Missing={sorted(missing)}, Extra={sorted(extra)}"
        )

    pool = list(state["unlabeled_pool"])
    labeled_train = list(state["labeled_train"])
    history = list(state.get("selected_history", []))

    for path in selected_paths:
        if path not in pool:
            raise ValueError(
                f"{path} is not in the current unlabeled pool. "
                "State and selection files may not match."
            )

        labeled_train.append({
            "path": path,
            "label": labels[path],
            "added_cycle": args.cycle,
        })

        pool.remove(path)

        history.append({
            "cycle": args.cycle,
            "path": path,
            "label": labels[path],
        })

    next_state = {
        "cycle": args.cycle,
        "labeled_train": labeled_train,
        "unlabeled_pool": pool,
        "selected_history": history,
    }

    output_path = STATES_DIR / f"cycle_{args.cycle:02d}.json"
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. "
            "Delete it manually only if you intentionally want to redo this cycle."
        )

    STATES_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(next_state, f, indent=2)

    print(f"Created {output_path}")
    print(f"  cycle          : {args.cycle}")
    print(f"  labeled_train  : {len(labeled_train)}")
    print(f"  unlabeled_pool : {len(pool)}")
    print(f"  added this cycle: {len(selected_paths)}")
    print("  rejected candidates remain in the pool.")


if __name__ == "__main__":
    main()