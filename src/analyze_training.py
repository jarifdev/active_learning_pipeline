#!/usr/bin/env python3
"""Read-only training-log analysis for ResNet or ViT (including FT2).

Accepts the CSV schema emitted by the existing training scripts. Finds the
minimum-validation-loss epoch and plots loss/accuracy curves. A heuristic
warning is not a diagnosis of overfitting. No models or datasets are changed.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = ("epoch", "train_loss", "val_loss", "train_acc", "val_acc")
OPTIONAL = ("val_defect_recall", "val_fnr", "val_defect_f1", "val_balanced_accuracy")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_log(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = set(REQUIRED) - set(fields)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
        rows = []
        seen = set()
        for line, row in enumerate(reader, 2):
            epoch = int(row["epoch"])
            if epoch < 1 or epoch in seen:
                raise ValueError(f"Invalid/repeated epoch {epoch} at line {line}")
            seen.add(epoch)
            parsed = {"epoch": epoch}
            for key in REQUIRED[1:]:
                value = float(row[key])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {key} at line {line}")
                parsed[key] = value
            if min(parsed["train_loss"], parsed["val_loss"]) < 0:
                raise ValueError(f"Negative loss at line {line}")
            for key in ("train_acc", "val_acc"):
                if not 0 <= parsed[key] <= 1:
                    raise ValueError(f"{key} must be in [0,1] at line {line}")
            for key in OPTIONAL:
                if key in row and row[key] not in (None, ""):
                    value = float(row[key])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"{key} must be finite and in [0,1] at line {line}")
                    parsed[key] = value
            rows.append(parsed)
    if not rows:
        raise ValueError("Empty training log")
    rows.sort(key=lambda r: r["epoch"])
    return rows


def summarize(name, path, rows, patience, min_delta):
    best = min(rows, key=lambda r: (r["val_loss"], r["epoch"]))
    last = rows[-1]
    warnings = []
    if rows[0]["epoch"] != 1 or any(b["epoch"] != a["epoch"] + 1 for a, b in zip(rows, rows[1:])):
        warnings.append("Epoch sequence is incomplete; inspect the log for missing epochs.")
    # A heuristic: a sustained worsening of validation loss while training loss
    # improves. It does not use test/shadow results and does not select an epoch.
    worsening = 0
    for prev, cur in zip(rows, rows[1:]):
        if cur["val_loss"] > prev["val_loss"] + min_delta and cur["train_loss"] < prev["train_loss"] - min_delta:
            worsening += 1
        else:
            worsening = 0
        if worsening >= patience:
            warnings.append(f"Possible overfitting pattern: {patience} consecutive epochs with rising validation loss and falling training loss (ending at epoch {cur['epoch']}).")
            break
    if last["epoch"] > best["epoch"] and last["val_loss"] > best["val_loss"] + min_delta:
        warnings.append("Final validation loss exceeds its minimum; the best-validation checkpoint may generalize better than the final epoch.")
    if best["train_acc"] > best["val_acc"] + 0.10:
        warnings.append("Training/validation accuracy gap exceeds 10 percentage points at the best epoch; investigate generalization.")
    if best["epoch"] == last["epoch"]:
        warnings.append("Best validation loss occurs at the epoch cap; a longer run could be investigated using validation data only, but is not automatically required.")
    return dict(name=name, log=str(path), log_sha256=sha256(path), epochs=len(rows), first_epoch=rows[0]["epoch"],
                last_epoch=last["epoch"], best_epoch=best["epoch"], best_val_loss=best["val_loss"],
                train_loss_at_best=best["train_loss"], val_loss_at_last=last["val_loss"],
                train_acc_at_best=best["train_acc"], val_acc_at_best=best["val_acc"],
                loss_gap_at_best=best["val_loss"]-best["train_loss"],
                accuracy_gap_at_best=best["train_acc"]-best["val_acc"], warnings=warnings,
                optional_metrics_at_best={k: best[k] for k in OPTIONAL if k in best})


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "run"


def plot_metric(plt, rows_by_name, metric, title, ylabel, out):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    found = False
    for name, rows in rows_by_name.items():
        values = [(r["epoch"], r[metric]) for r in rows if metric in r]
        if values:
            ax.plot([v[0] for v in values], [v[1] for v in values], marker=".", label=name)
            found = True
    if not found:
        plt.close(fig)
        return False
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--run", action="append", required=True, metavar="NAME=CSV", help="A training log; repeat for baseline, active, random")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=1e-4)
    args = p.parse_args(argv)
    if args.patience < 1 or args.min_delta < 0:
        p.error("--patience must be >=1 and --min-delta must be nonnegative")
    root = args.root.resolve()
    out = args.output or root / "results" / "audits" / ("training_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    out = out if out.is_absolute() else root / out
    runs, summaries, errors = {}, [], []
    for spec in args.run:
        if "=" not in spec:
            p.error("Each --run must have the form NAME=CSV")
        name, raw = spec.split("=", 1)
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        name = name.strip()
        if not name or name in runs:
            p.error("Run names must be nonempty and unique")
        try:
            rows = read_log(path)
            runs[name] = rows
            summaries.append(summarize(name, path, rows, args.patience, args.min_delta))
        except (OSError, ValueError, KeyError) as e:
            errors.append(f"{name}: {e}")
    if not runs:
        for error in errors:
            print("ERROR:", error, file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. CSV/JSON summaries will still be saved.")
        plt = None
    if plt is not None:
        for name, rows in runs.items():
            base = safe_name(name)
            plot_metric(plt, {"Training": rows, "Validation": [dict(r, train_loss=r["val_loss"]) for r in rows]}, "train_loss", f"{name}: loss", "Cross-entropy loss", out / f"{base}_loss.png")
            plot_metric(plt, {"Training": rows, "Validation": [dict(r, train_acc=r["val_acc"]) for r in rows]}, "train_acc", f"{name}: accuracy", "Accuracy", out / f"{base}_accuracy.png")
        for metric, title, ylabel in (("val_loss", "Validation loss", "Cross-entropy loss"), ("val_acc", "Validation accuracy", "Accuracy"),
                                      ("val_defect_recall", "Validation defect recall", "Recall"), ("val_fnr", "Validation false-negative rate", "FNR"),
                                      ("val_defect_f1", "Validation defect F1", "F1")):
            plot_metric(plt, runs, metric, title, ylabel, out / f"comparison_{metric}.png")
    fields = ["name", "log", "log_sha256", "epochs", "first_epoch", "last_epoch", "best_epoch", "best_val_loss", "train_loss_at_best", "val_loss_at_last", "train_acc_at_best", "val_acc_at_best", "loss_gap_at_best", "accuracy_gap_at_best", "warnings"]
    write_csv(out / "summary.csv", [{**s, "warnings": "; ".join(s["warnings"])} for s in summaries], fields)
    all_rows = []
    for name, rows in runs.items():
        all_rows.extend(dict(run=name, **row) for row in rows)
    write_csv(out / "all_epochs.csv", all_rows, ["run", *REQUIRED, *OPTIONAL])
    report = dict(summaries=summaries, errors=errors, notes=[
        "Best epoch is determined by minimum validation loss, not test/shadow performance.",
        "Training metrics may include random augmentation; compare with deterministic training-set evaluation if needed.",
        "Warnings are heuristics, not proof of overfitting or absence of overfitting.",
        "Weights-only checkpoints do not prove which epoch, split, or configuration produced them.",
        "The script does not retrain, select a new checkpoint, or modify experiment inputs."])
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nTraining-log summary")
    for s in summaries:
        print(f"{s['name']}: best epoch {s['best_epoch']}, val_loss={s['best_val_loss']:.4f}, train_acc={s['train_acc_at_best']:.4f}, val_acc={s['val_acc_at_best']:.4f}")
        for warning in s["warnings"]:
            print("  WARNING:", warning)
    for error in errors:
        print("ERROR:", error)
    print("Reports:", out)
    if not all(any(k in r for r in rows) for rows in runs.values() for k in ("val_defect_recall", "val_fnr", "val_defect_f1")):
        print("Note: current logs contain no per-class validation metrics; recall/FNR curves require future logging or saved per-epoch checkpoints.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
