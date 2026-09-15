#!/usr/bin/env python3
"""
init_resnet_state.py

Creates the Cycle-0 states for both the ResNet ACTIVE and RANDOM branches
directly from split.json.

Creates:
    states/cycle_00.json
    states/random/cycle_00.json

Both branches start from exactly the same:
    - original labeled training split
    - original unlabeled pool

Run once:
    python src/init_resnet_state.py
"""

from pathlib import Path
import argparse
import json

from dataset import CLASS_TO_LABEL


MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
SPLIT_PATH = PROJECT_ROOT / "split.json"



def build_cycle_zero():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH} does not exist. Run make_split.py first."
        )

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    if "train" not in split or "unlabeled_pool" not in split:
        raise ValueError(
            "split.json must contain 'train' and 'unlabeled_pool'."
        )

    labeled_train = []

    for rel_path in split["train"]:
        class_name = rel_path.split("/")[0]

        if class_name not in CLASS_TO_LABEL:
            raise ValueError(
                f"Unknown training class in path: {rel_path}"
            )

        labeled_train.append({
            "path": rel_path,
            "label": CLASS_TO_LABEL[class_name],
            "added_cycle": 0,
        })

    state = {
        "cycle": 0,
        "labeled_train": labeled_train,
        "unlabeled_pool": list(split["unlabeled_pool"]),
        "selected_history": [],
    }

    if len(state["labeled_train"]) != 160:
        print(
            f"WARNING: expected 160 initial training images, "
            f"found {len(state['labeled_train'])}."
        )

    if len(state["unlabeled_pool"]) != 200:
        print(
            f"WARNING: expected 200 initial pool images, "
            f"found {len(state['unlabeled_pool'])}."
        )

    return state


def write_state(path, state):
    if path.exists():
        raise FileExistsError(
            f"{path} already exists.\n"
            "Delete it manually only if you intentionally want to restart "
            "the ResNet experiment."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["old_pipeline", "new_pipeline"], required=True)
    args = parser.parse_args()

    ACTIVE_PATH = PROJECT_ROOT / "states" / "resnet" / args.pipeline / "active" / "cycle_00.json"
    RANDOM_PATH = PROJECT_ROOT / "states" / "resnet" / args.pipeline / "random" / "cycle_00.json"

    cycle_zero = build_cycle_zero()

    # Write independent files containing the exact same Cycle-0 data.
    active_state = json.loads(json.dumps(cycle_zero))
    random_state = json.loads(json.dumps(cycle_zero))

    write_state(ACTIVE_PATH, active_state)
    write_state(RANDOM_PATH, random_state)

    print("ResNet Cycle-0 states created.")
    print(f"Active : {ACTIVE_PATH}")
    print(f"Random : {RANDOM_PATH}")
    print(f"Labeled train : {len(cycle_zero['labeled_train'])}")
    print(f"Unlabeled pool: {len(cycle_zero['unlabeled_pool'])}")
    print("\nBoth branches start from identical Cycle-0 data.")


if __name__ == "__main__":
    main()
