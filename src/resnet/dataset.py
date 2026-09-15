"""
dataset.py

Contains three PyTorch Dataset classes:

1. DefectDataset
   Used for the ORIGINAL labeled splits:
   train / val / test / shadow / spare

2. ActiveLearningDataset
   Used for the GROWING labeled training set during active learning.

3. UnlabeledDataset
   Used for the CURRENT unlabeled pool.
"""

import os
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# 0 = no_defect
# 1 = defect
MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
SPLIT_JSON = PROJECT_ROOT / "split.json"
DATA_DIR = PROJECT_ROOT / "raw_data"


CLASS_TO_LABEL = {
    "no_defect": 0,
    "defect": 1,
}


# ============================================================
# 1. ORIGINAL LABELED DATASET
# ============================================================

class DefectDataset(Dataset):

    def __init__(self, split_json, split_name, data_dir, transform=None):

        with open(split_json, "r") as f:
            split = json.load(f)

        if split_name not in split:
            raise ValueError(
                f"'{split_name}' not found in {split_json}. "
                f"Available keys: {list(split.keys())}"
            )

        self.filepaths = split[split_name]
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):

        rel_path = self.filepaths[idx]

        # Example:
        # "defect/img001.bmp"
        # -> "defect"
        class_name = rel_path.split("/")[0]

        label = CLASS_TO_LABEL[class_name]

        full_path = os.path.join(
            self.data_dir,
            rel_path
        )

        image = Image.open(full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# 2. ACTIVE-LEARNING TRAINING DATASET
# ============================================================

class ActiveLearningDataset(Dataset):
    """
    Used after active learning starts.

    Unlike DefectDataset, the label is NOT determined from
    the image folder.

    Instead, each item explicitly contains:

        {
            "path": "unlabeled_pool/img500.bmp",
            "label": 1
        }

    This is necessary because newly labeled images remain
    physically inside unlabeled_pool/.
    """

    def __init__(self, samples, data_dir, transform=None):

        self.samples = samples
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        rel_path = sample["path"]
        label = sample["label"]

        full_path = os.path.join(
            self.data_dir,
            rel_path
        )

        image = Image.open(full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# 3. CURRENT UNLABELED POOL
# ============================================================

class UnlabeledDataset(Dataset):
    """
    Used for inference on the CURRENT unlabeled pool.

    We pass the current list of filenames directly instead
    of rereading split.json.

    Example:

        Cycle 0 -> 200 filenames
        Cycle 1 -> 180 filenames
        Cycle 2 -> 160 filenames
    """

    def __init__(self, filepaths, data_dir, transform=None):

        self.filepaths = filepaths
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):

        rel_path = self.filepaths[idx]

        full_path = os.path.join(
            self.data_dir,
            rel_path
        )

        image = Image.open(full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, rel_path


# ============================================================
# QUICK MANUAL TEST
# ============================================================

if __name__ == "__main__":

    from transforms import (
        get_train_transforms,
        get_eval_transforms
    )

    # --------------------------------------------------------
    # Test original training dataset
    # --------------------------------------------------------

    train_ds = DefectDataset(
        SPLIT_JSON,
        "train",
        DATA_DIR,
        get_train_transforms()
    )

    print(f"Original train size: {len(train_ds)}")

    img, label = train_ds[0]

    print(
        f"First original train item -> "
        f"shape: {img.shape}, label: {label}"
    )


    # --------------------------------------------------------
    # Test original unlabeled pool from split.json
    #
    # This part is ONLY for testing.
    # During active learning the list will come from cycle state.
    # --------------------------------------------------------

    with open(SPLIT_JSON, "r") as f:
        split = json.load(f)

    pool_ds = UnlabeledDataset(
        split["unlabeled_pool"],
        DATA_DIR,
        get_eval_transforms()
    )

    print(f"Initial unlabeled pool size: {len(pool_ds)}")

    img, fname = pool_ds[0]

    print(
        f"First pool item -> "
        f"shape: {img.shape}, filename: {fname}"
    )


    # --------------------------------------------------------
    # Small fake test of ActiveLearningDataset
    # --------------------------------------------------------

    fake_samples = [
        {
            "path": split["train"][0],
            "label": (
                1
                if split["train"][0].startswith("defect/")
                else 0
            )
        }
    ]

    al_ds = ActiveLearningDataset(
        fake_samples,
        DATA_DIR,
        get_train_transforms()
    )

    img, label = al_ds[0]

    print(
        f"ActiveLearningDataset test -> "
        f"shape: {img.shape}, label: {label}"
    )