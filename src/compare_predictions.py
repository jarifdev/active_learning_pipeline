"""
Paired, deterministic evaluation of existing Aachen checkpoints.
Run from the project root. No training, selection, or checkpoint writes.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def named_paths(values, root):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Expected name=path: " + value)
        name, path = value.split("=", 1)
        if not name or name in result:
            raise ValueError("Empty or repeated model name: " + name)
        p = Path(path)
        result[name] = p if p.is_absolute() else root / p
    return result


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def records_from_split(split, split_name):
    records = []
    for path in split[split_name]:
        path = str(path).replace("\\", "/")
        label_name = path.split("/")[0]
        if label_name not in ("defect", "no_defect"):
            raise ValueError("Cannot infer label from " + path)
        records.append((path, int(label_name == "defect")))
    return records


def records_from_state(path):
    state = json.loads(path.read_text(encoding="utf-8"))
    return [(str(x["path"]).replace("\\", "/"), int(x["label"]))
            for x in state["labeled_train"]]


class AuditDataset(Dataset):
    def __init__(self, records, data_dir, transform):
        self.records = records
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(self.data_dir / path) as image:
            image = self.transform(image.convert("RGB"))
        return image, label, path


def build_model(architecture):
    from torchvision.models import resnet50, vit_b_16
    if architecture == "resnet":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif architecture == "vit-ft2":
        # Uses the exact uploaded FT2 builder, without downloading weights.
        builder = importlib.import_module("model_vit_ft2")
        model = builder.build_vit(
            freeze_backbone=True, pretrained=False, train_last_n_blocks=2)
    else:
        raise ValueError(architecture)
    return model


def load_model(architecture, checkpoint, device):
    model = build_model(architecture)
    # Load only checkpoints you trust. Existing project files store state_dicts.
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    if not isinstance(state, dict):
        raise TypeError("Expected a state_dict checkpoint.")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, records, data_dir, transform, device, batch_size):
    loader = DataLoader(AuditDataset(records, data_dir, transform),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    result = {}
    for images, labels, paths in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
        predictions = probabilities.argmax(dim=1)
        for path, label, pred, prob in zip(paths, labels, predictions, probabilities):
            true = int(label)
            predicted = int(pred)
            category = ("FN" if true == 1 and predicted == 0 else
                        "FP" if true == 0 and predicted == 1 else
                        "TP" if true == 1 else "TN")
            result[path] = {
                "true_label": true, "predicted_label": predicted,
                "p_no_defect": float(prob[0]), "p_defect": float(prob[1]),
                "result": category,
            }
    return result


def metrics(predictions):
    tn = fp = fn = tp = 0
    for row in predictions.values():
        if row["result"] == "TN": tn += 1
        elif row["result"] == "FP": fp += 1
        elif row["result"] == "FN": fn += 1
        else: tp += 1
    n = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"n": n, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tn + tp) / n if n else 0,
            "defect_precision": precision, "defect_recall": recall,
            "defect_f1": 2 * precision * recall / (precision + recall)
                         if precision + recall else 0,
            "defect_fnr": fn / (fn + tp) if fn + tp else 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--architecture", choices=["resnet", "vit-ft2"], required=True)
    p.add_argument("--model", action="append", required=True, help="name=checkpoint.pt")
    p.add_argument("--split", action="append", choices=["val", "test", "shadow", "train"],
                   required=True)
    p.add_argument("--train-state", action="append", default=[], help="name=cycle_05.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src" / "vit"))
    sys.path.insert(0, str(root / "src"))
    from transforms import get_eval_transforms
    checkpoints = named_paths(args.model, root)
    train_states = named_paths(args.train_state, root)
    for path in list(checkpoints.values()) + list(train_states.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    split_path = root / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    output = args.out if args.out.is_absolute() else root / args.out
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device)
    manifest = {"architecture": args.architecture, "device": str(device),
                "split_sha256": digest(split_path), "checkpoints": {},
                "train_states": {}, "threshold": 0.5,
                "notes": "Argmax decisions; original checkpoints and data unchanged."}
    all_metrics = []
    for name, path in train_states.items():
        manifest["train_states"][name] = {"path": str(path), "sha256": digest(path)}
    transform = get_eval_transforms()
    for name, checkpoint in checkpoints.items():
        print("Loading", name, checkpoint, flush=True)
        manifest["checkpoints"][name] = {"path": str(checkpoint),
                                           "sha256": digest(checkpoint)}
    for split_name in dict.fromkeys(args.split):
        base = records_from_split(split, split_name)
        if split_name != "train":
            canonical = dict(base)
        results = {}
        for name, checkpoint in checkpoints.items():
            if split_name == "train" and name in train_states:
                records = records_from_state(train_states[name])
            else:
                records = base
            if len(records) != len(set(path for path, _ in records)):
                raise ValueError("Duplicate paths in " + name + "/" + split_name)
            if any(label not in (0, 1) for _, label in records):
                raise ValueError("Invalid binary label")
            if split_name != "train" and dict(records) != canonical:
                raise ValueError("Evaluation records differ across models")
            model = load_model(args.architecture, checkpoint, device)
            preds = predict(model, records, root / "raw_data", transform,
                            device, args.batch_size)
            del model
            results[name] = preds
            m = metrics(preds)
            all_metrics.append({"split": split_name, "model": name, **m})
            rows = [{"filename": path, **row} for path, row in sorted(preds.items())]
            write_csv(output / f"{split_name}_{name}_predictions.csv", rows,
                      ["filename", "true_label", "predicted_label",
                       "p_no_defect", "p_defect", "result"])
            print(f"{split_name} {name}: accuracy={m['accuracy']:.4f}, "
                  f"FN={m['fn']}, FP={m['fp']}", flush=True)
        if split_name == "train":
            # Training sets differ; do not invent a paired comparison.
            continue
        columns = ["filename", "true_label"]
        for name in checkpoints:
            columns += [f"{name}_pred", f"{name}_p_defect", f"{name}_result"]
        columns += ["wrong_models", "fn_models"]
        merged = []
        for path, label in base:
            row = {"filename": path, "true_label": label}
            wrong, fns = [], []
            for name in checkpoints:
                r = results[name][path]
                row[f"{name}_pred"] = r["predicted_label"]
                row[f"{name}_p_defect"] = r["p_defect"]
                row[f"{name}_result"] = r["result"]
                if r["result"] in ("FN", "FP"):
                    wrong.append(name)
                if r["result"] == "FN":
                    fns.append(name)
            row["wrong_models"] = ";".join(wrong)
            row["fn_models"] = ";".join(fns)
            merged.append(row)
        write_csv(output / f"{split_name}_paired.csv", merged, columns)
        write_csv(output / f"{split_name}_errors.csv",
                  [r for r in merged if r["wrong_models"]], columns)
        write_csv(output / f"{split_name}_false_negatives.csv",
                  [r for r in merged if r["fn_models"]], columns)
        print(f"Paired audit: {sum(bool(r['fn_models']) for r in merged)} "
              f"unique missed-defect images on {split_name}.", flush=True)
    columns = ["split", "model", "n", "tn", "fp", "fn", "tp",
               "accuracy", "defect_precision", "defect_recall",
               "defect_f1", "defect_fnr"]
    write_csv(output / "metrics.csv", all_metrics, columns)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved:", output)


if __name__ == "__main__":
    main()
