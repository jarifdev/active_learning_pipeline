"""
init_vit_state.py

Creates independent Cycle-0 states for the ViT replication directly from
the existing split.json.

Creates:
    states/vit/active/cycle_00.json
    states/vit/random/cycle_00.json

Both branches start from exactly the same original:
    - labeled training split
    - unlabeled pool

Run once:
    python src/init_vit_state.py
"""

from pathlib import Path
import argparse
import json

MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
SPLIT_JSON = PROJECT_ROOT / "split.json"


CLASS_TO_LABEL = {
    "no_defect": 0,
    "defect": 1,
}


def build_cycle_zero(strategy):
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
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
                f"Cannot infer class from training path: {rel_path}"
            )

        labeled_train.append({
            "path": rel_path,
            "label": CLASS_TO_LABEL[class_name],
        })

    return {
        "cycle": 0,
        "labeled_train": labeled_train,
        "unlabeled_pool": list(split["unlabeled_pool"]),
        "selected_history": [],
        "strategy": strategy,
        "model": "vit_b_16",
    }


def write_state(path, state):
    if path.exists():
        raise FileExistsError(
            f"{path} already exists.\n"
            "Delete it manually only if you intentionally want to restart "
            "the ViT experiment."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["old_pipeline", "new_pipeline"], required=True)
    args = parser.parse_args()
    ACTIVE_DIR = PROJECT_ROOT / "states" / "vit" / args.pipeline / "active"
    RANDOM_DIR = PROJECT_ROOT / "states" / "vit" / args.pipeline / "random"

    active_state = build_cycle_zero("active")
    random_state = build_cycle_zero("random")

    active_path = ACTIVE_DIR / "cycle_00.json"
    random_path = RANDOM_DIR / "cycle_00.json"

    write_state(active_path, active_state)
    write_state(random_path, random_state)

    print("ViT Cycle-0 states created.")
    print(f"Active : {active_path}")
    print(f"Random : {random_path}")
    print(f"Labeled train : {len(active_state['labeled_train'])}")
    print(f"Unlabeled pool: {len(active_state['unlabeled_pool'])}")


if __name__ == "__main__":
    main()
