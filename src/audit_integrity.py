#!/usr/bin/env python3
"""Read-only integrity audit for the Aachen active-learning experiment.

Checks the exact split/state schema used by dataset.py and init_resnet_state.py.
No training, model loading, relabeling, or source-file modifications occur.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

FIXED = ("train", "val", "test", "shadow", "spare", "unlabeled_pool", "pool_spare")
LABELS = {"no_defect": 0, "defect": 1}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def canonical(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image path must be a nonempty string")
    value = value.replace("\\", "/")
    p = PurePosixPath(value)
    if not p.parts or p.is_absolute() or ":" in p.parts[0] or any(x in (".", "..") for x in value.split("/")):
        raise ValueError("absolute/traversal paths are not allowed")
    return p.as_posix()


def label(value):
    if isinstance(value, bool) or str(value).strip() not in ("0", "1"):
        raise ValueError("label must be 0 or 1")
    return int(value)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class Audit:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.data = root / "raw_data"
        self.issues, self.states, self.groups = [], {}, {}
        self.paths = set()
        self.label_registry = defaultdict(list)
        self.manifest, self.duplicates, self.near = [], [], []
        self.original_pool = set()
        self.initial_train = {}

    def issue(self, severity, code, message, **details):
        self.issues.append(dict(severity=severity, code=code, message=message, **details))

    def check(self, condition, code, message, **details):
        if not condition:
            self.issue("ERROR", code, message, **details)

    def path(self, value, origin):
        try:
            p = canonical(value)
            self.paths.add(p)
            return p
        except (ValueError, TypeError) as e:
            self.issue("ERROR", "INVALID_PATH", str(e), origin=origin, value=str(value))
            return None

    def entries(self, values, origin, explicit=False):
        result = {}
        if not isinstance(values, list):
            self.issue("ERROR", "BAD_SCHEMA", "Expected a list", origin=origin)
            return result
        for index, item in enumerate(values):
            where = f"{origin}[{index}]"
            try:
                if isinstance(item, dict):
                    if "path" not in item:
                        raise ValueError("missing path")
                    raw, raw_label = item["path"], item.get("label")
                else:
                    raw, raw_label = item, None
                p = self.path(raw, where)
                if p is None:
                    continue
                if explicit:
                    if raw_label is None:
                        raise ValueError("missing explicit label")
                    y = label(raw_label)
                elif raw_label is not None:
                    y = label(raw_label)
                else:
                    y = LABELS.get(p.split("/")[0])
                if p in result:
                    self.issue("ERROR", "DUPLICATE_PATH", "Repeated image in one collection", origin=origin, path=p)
                elif explicit and y is None:
                    raise ValueError("missing label")
                result[p] = y
                if y is not None:
                    self.label_registry[p].append((origin, y))
            except (ValueError, TypeError) as e:
                self.issue("ERROR", "BAD_ENTRY", str(e), origin=where)
        return result

    def read_split(self):
        path = self.root / "split.json"
        if not path.is_file():
            self.issue("ERROR", "MISSING_SPLIT", "split.json not found", path=str(path))
            return
        self.split_hash = digest(path.read_bytes())
        split = read_json(path)
        if not isinstance(split, dict):
            self.issue("ERROR", "BAD_SCHEMA", "split.json must be an object")
            return
        for key in FIXED:
            if key not in split:
                if key in ("spare", "pool_spare"):
                    continue
                self.issue("ERROR", "MISSING_SPLIT_KEY", f"Missing split: {key}")
                continue
            self.groups[key] = self.entries(split[key], f"split.{key}")
        for key, expected in (("train", self.args.initial_train), ("val", self.args.val_size),
                              ("test", self.args.test_size), ("shadow", self.args.shadow_size),
                              ("unlabeled_pool", self.args.initial_pool)):
            if key in self.groups:
                self.check(len(self.groups[key]) == expected, "SPLIT_SIZE", f"{key}: expected {expected}, got {len(self.groups[key])}")
        owners = defaultdict(list)
        for group, values in self.groups.items():
            for p in values:
                owners[p].append(group)
        for p, groups in owners.items():
            if len(groups) > 1:
                self.issue("ERROR", "SPLIT_OVERLAP", "Image assigned to multiple split groups", path=p, groups=groups)
        self.initial_train = self.groups.get("train", {})
        self.original_pool = set(self.groups.get("unlabeled_pool", {}))
        for p, y in self.initial_train.items():
            expected = LABELS.get(p.split("/")[0])
            self.check(y == expected, "TRAIN_LABEL", f"Invalid initial label: {p}", path=p)

    def read_state(self, folder, cycle):
        path = folder / f"cycle_{cycle:02d}.json"
        if not path.exists():
            self.issue("ERROR", "MISSING_STATE", "Cycle state is missing", path=str(path))
            return None
        try:
            state = read_json(path)
            if not isinstance(state, dict):
                raise ValueError("state must be a JSON object")
            if state.get("cycle", cycle) != cycle:
                self.issue("ERROR", "CYCLE_MISMATCH", "State cycle metadata differs from filename", path=str(path))
            train = self.entries(state["labeled_train"], f"{folder}:cycle_{cycle:02d}.labeled_train", True)
            pool = self.entries(state["unlabeled_pool"], f"{folder}:cycle_{cycle:02d}.unlabeled_pool")
            return dict(train=train, pool=set(pool), hash=digest(path.read_bytes()), path=str(path))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            self.issue("ERROR", "BAD_STATE", str(e), path=str(path))
            return None

    def audit_branch(self, folder):
        name = folder.relative_to(self.root).as_posix() if folder.is_relative_to(self.root) else str(folder)
        print(f"Checking {name}")
        records = []
        prev = None
        for cycle in range(self.args.cycles + 1):
            current = self.read_state(folder, cycle)
            if current is None:
                prev = None
                continue
            t, p = current["train"], current["pool"]
            expected_train = self.args.initial_train + cycle * self.args.per_cycle
            expected_pool = self.args.initial_pool - cycle * self.args.per_cycle
            self.check(len(t) == expected_train, "STATE_TRAIN_SIZE", f"{name} C{cycle}: expected {expected_train} train, got {len(t)}")
            self.check(len(p) == expected_pool, "STATE_POOL_SIZE", f"{name} C{cycle}: expected {expected_pool} pool, got {len(p)}")
            self.check(not (set(t) & p), "TRAIN_POOL_OVERLAP", f"{name} C{cycle}: training and pool overlap")
            self.check(set(t) | p == set(self.initial_train) | self.original_pool,
                       "STATE_UNIVERSE", f"{name} C{cycle}: training/pool universe differs from Cycle 0 split")
            for group in ("val", "test", "shadow", "spare", "pool_spare"):
                bad = (set(t) | p) & set(self.groups.get(group, {}))
                if bad:
                    self.issue("ERROR", "HELDOUT_LEAKAGE", f"{name} C{cycle} overlaps {group}", paths=sorted(bad)[:30], count=len(bad))
            for path, y in self.initial_train.items():
                self.check(path in t and t.get(path) == y, "INITIAL_TRAIN_CHANGED", f"{name} C{cycle}: original training image missing/relabelled: {path}")
            for path in t:
                if path not in self.initial_train and path not in self.original_pool:
                    self.issue("ERROR", "UNAUTHORIZED_TRAIN_IMAGE", "Acquired image was not in original pool", branch=name, cycle=cycle, path=path)
            if cycle == 0:
                self.check(t == self.initial_train, "C0_TRAIN_MISMATCH", f"{name}: initial train does not match split.json")
                self.check(p == self.original_pool, "C0_POOL_MISMATCH", f"{name}: initial pool does not match split.json")
            if prev is not None:
                added = set(t) - set(prev["train"])
                removed = set(prev["train"]) - set(t)
                removed_pool = prev["pool"] - p
                added_pool = p - prev["pool"]
                self.check(len(added) == self.args.per_cycle, "ACQUISITION_COUNT", f"{name} C{cycle}: added {len(added)} instead of {self.args.per_cycle}")
                self.check(not removed, "REMOVED_TRAIN", f"{name} C{cycle}: training images disappeared", paths=sorted(removed))
                self.check(removed_pool == added and not added_pool, "POOL_CONSERVATION", f"{name} C{cycle}: pool changes do not match acquired images")
                for path in set(prev["train"]) & set(t):
                    self.check(prev["train"][path] == t[path], "RELABELLED_IMAGE", f"{name} C{cycle}: label changed for {path}")
                self.check(len(t) + len(p) == len(prev["train"]) + len(prev["pool"]), "UNIVERSE_SIZE", f"{name} C{cycle}: total image count changed")
                self.check(set(t) == set(prev["train"]) | added, "TRAIN_LINEAGE", f"{name} C{cycle}: invalid training progression")
                self.audit_selection(folder, cycle, added)
            counts = Counter(t.values())
            records.append(dict(branch=name, cycle=cycle, train=len(t), pool=len(p), defects=counts[1], no_defects=counts[0], state_hash=current["hash"]))
            prev = current
        self.states[name] = records
        return records

    def audit_selection(self, folder, cycle, added):
        # Selection folders mirror states folders in the original project.
        rel = folder.relative_to(self.root / "states")

        selection_mapping = {
            "random_new": "random",
            "vit/active_new": "vit/active",
            "vit/random_new": "vit/random",
        }

        rel_str = rel.as_posix()
        selection_rel = selection_mapping.get(rel_str, rel_str)

        csv_path = (
            self.root
            / "selections"
            / selection_rel
            / f"cycle_{cycle:02d}_to_label.csv"
        )
        if not csv_path.exists():
            self.issue("WARNING", "MISSING_SELECTION_CSV", "Cannot verify the saved selection against the state", path=str(csv_path))
            return
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        selected = self.entries([{"path": r.get("filename"), "label": r.get("label") or None} for r in rows], str(csv_path), True)
        self.check(set(selected) == added, "SELECTION_MISMATCH", f"Cycle {cycle}: selected CSV does not match newly acquired images", path=str(csv_path))
        self.check(len(rows) == self.args.per_cycle, "SELECTION_SIZE", f"Cycle {cycle}: selection CSV has {len(rows)} rows", path=str(csv_path))
        state = self.read_state(folder, cycle)
        if state:
            for p, y in selected.items():
                if y is not None:
                    self.check(state["train"].get(p) == y, "SELECTION_LABEL_MISMATCH", f"Cycle {cycle}: selection label differs from state for {p}")

    def check_labels(self):
        for p, observations in self.label_registry.items():
            values = {y for _, y in observations}
            if len(values) > 1:
                self.issue("ERROR", "CONFLICTING_LABELS", "Same image has different labels", path=p, observations=observations)

    def inspect_images(self):
        try:
            from PIL import Image, ImageOps
        except ImportError:
            self.issue("ERROR", "PIL_MISSING", "Install Pillow to verify images and compute decoded hashes")
            return
        by_bytes, by_pixels, hashes = defaultdict(list), defaultdict(list), {}
        casefold = defaultdict(list)
        for p in sorted(self.paths):
            casefold[p.casefold()].append(p)
            file = self.data / p
            row = dict(path=p, exists=file.is_file(), bytes_sha256="", pixels_sha256="", dhash="", width="", height="")
            if not file.resolve().is_relative_to(self.data.resolve()):
                self.issue("ERROR", "PATH_ESCAPE", "Image resolves outside raw_data", path=p)
                self.manifest.append(row)
                continue
            if not file.is_file():
                self.issue("ERROR", "MISSING_IMAGE", "Image file not found", path=p)
                self.manifest.append(row)
                continue
            try:
                raw_hash = digest(file.read_bytes())
                with Image.open(file) as img:
                    img.load()
                    rgb = ImageOps.exif_transpose(img).convert("RGB")
                    pixel_hash = digest(f"{rgb.size}|RGB".encode() + rgb.tobytes())
                    small = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                    px = list(small.getdata())
                    bits = 0
                    for y in range(8):
                        for x in range(8):
                            bits = (bits << 1) | int(px[y * 9 + x] > px[y * 9 + x + 1])
                    row.update(bytes_sha256=raw_hash, pixels_sha256=pixel_hash, dhash=f"{bits:016x}", width=rgb.width, height=rgb.height)
                    hashes[p] = bits
                by_bytes[raw_hash].append(p)
                by_pixels[pixel_hash].append(p)
            except Exception as e:
                self.issue("ERROR", "UNREADABLE_IMAGE", str(e), path=p)
            self.manifest.append(row)
        for group in casefold.values():
            if len(group) > 1:
                self.issue("WARNING", "CASE_COLLISION", "Paths differ only by case", paths=group)
        for kind, groups in (("bytes", by_bytes), ("decoded_pixels", by_pixels)):
            for h, paths in groups.items():
                if len(paths) > 1:
                    self.duplicates.append(dict(kind=kind, hash=h, paths=paths))
        # A repeated original image is especially serious across held-out/train groups.
        owners = defaultdict(set)
        for name, group in self.groups.items():
            for p in group:
                owners[p].add(name)
        for branch, records in self.states.items():
            for p in self.branch_final_train.get(branch, set()):
                owners[p].add(f"{branch}:train")
        for row in self.duplicates:
            paths = row["paths"]
            owner_sets = [owners.get(p, set()) for p in paths]
            heldout = {"val", "test", "shadow", "spare", "pool_spare"}
            cross = any(
                left != right and (a in heldout or b in heldout)
                for i, left in enumerate(owner_sets)
                for right in owner_sets[i + 1:]
                for a in left for b in right if a != b
            )
            if cross:
                self.issue("ERROR", "EXACT_CROSS_SPLIT_DUPLICATE", "Identical image content occurs across data groups", kind=row["kind"], paths=paths, groups=[sorted(s) for s in owner_sets])
            else:
                self.issue("WARNING", "EXACT_DUPLICATE", "Identical image content found; review redundancy", kind=row["kind"], paths=paths)
        if not self.args.near_duplicates:
            return
        if len(hashes) > self.args.near_limit:
            self.issue("WARNING", "NEAR_CHECK_SKIPPED", "Too many images for exhaustive near-duplicate comparison", count=len(hashes), limit=self.args.near_limit)
            return
        # dHash is a screening tool only: low-texture factory images can collide.
        items = sorted(hashes.items())
        protected = {"val", "test", "shadow", "spare", "pool_spare"}
        near_total = 0
        for i, (a, ha) in enumerate(items):
            for b, hb in items[i + 1:]:
                if owners.get(a, set()) == owners.get(b, set()):
                    continue
                if not ((owners.get(a, set()) | owners.get(b, set())) & protected):
                    continue
                distance = (ha ^ hb).bit_count()
                if distance <= self.args.near_threshold:
                    near_total += 1
                    if len(self.near) < self.args.max_near_pairs:
                        self.near.append(dict(path_a=a, path_b=b, hamming_distance=distance,
                                              groups_a=";".join(sorted(owners.get(a, set()))), groups_b=";".join(sorted(owners.get(b, set())))))
        if near_total:
            self.issue("WARNING", "NEAR_DUPLICATE_CANDIDATES", "Perceptual-hash matches need human review; they are not proof of leakage", count=near_total)
        if near_total > len(self.near):
            self.issue("WARNING", "NEAR_REPORT_TRUNCATED", f"Saved first {len(self.near)} of {near_total} candidate pairs; increase --max-near-pairs to save more")

    def run(self):
        self.branch_final_train = {}
        self.read_split()
        folders = [self.root / x for x in self.args.state_dir] if self.args.state_dir else [
            self.root / x for x in ("states", "states/random", "states/vit/active", "states/vit/random")
            if (self.root / x / "cycle_00.json").exists()]
        if not folders:
            self.issue("ERROR", "NO_STATES", "No Cycle-0 state folders found; specify --state-dir")
        for folder in folders:
            records = self.audit_branch(folder)
            if records:
                final = self.read_state(folder, self.args.cycles)
                if final:
                    name = records[0]["branch"]
                    self.branch_final_train[name] = set(final["train"])
        self.check_labels()
        self.inspect_images()
        self.compare_branches(folders)

    def compare_branches(self, folders):
        starts = []
        for folder in folders:
            p = folder / "cycle_00.json"
            if p.exists():
                s = read_json(p)
                train = self.entries(s.get("labeled_train"), str(p) + ":train", True)
                pool = self.entries(s.get("unlabeled_pool"), str(p) + ":pool")
                starts.append((str(folder), train, set(pool)))
        for name, train, pool in starts[1:]:
            self.check(train == starts[0][1] and pool == starts[0][2], "BRANCH_START_MISMATCH", f"{name} does not match first branch's Cycle-0 data")
        if len(starts) < 2:
            self.issue("WARNING", "ONE_BRANCH", "Only one branch is available; cross-branch equality cannot be checked")

    def save(self, out):
        out.mkdir(parents=True, exist_ok=False)
        write_csv(out / "issues.csv", self.issues, ["severity", "code", "message", "path", "origin", "branch", "cycle", "count", "paths", "groups", "observations"])
        rows = [r for branch in self.states.values() for r in branch]
        write_csv(out / "state_summary.csv", rows, ["branch", "cycle", "train", "pool", "defects", "no_defects", "state_hash"])
        write_csv(out / "image_manifest.csv", self.manifest, ["path", "exists", "bytes_sha256", "pixels_sha256", "dhash", "width", "height"])
        write_csv(out / "exact_duplicates.csv", self.duplicates, ["kind", "hash", "paths"])
        write_csv(out / "near_duplicates.csv", self.near, ["path_a", "path_b", "hamming_distance", "groups_a", "groups_b"])
        counts = Counter(x["severity"] for x in self.issues)
        report = dict(status="FAIL" if counts["ERROR"] else "PASS_WITH_WARNINGS" if counts["WARNING"] else "PASS",
                      split_sha256=getattr(self, "split_hash", None), counts=dict(counts),
                      issues=self.issues, branches=self.states, settings=vars(self.args),
                      notes=["Read-only audit; source files were not modified.",
                             "A clean audit does not prove absence of semantic leakage or overfitting.",
                             "Near-duplicate dHash matches require human review.",
                             "Only the current files were audited; missing old experiment histories cannot be reconstructed."])
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\n{report['status']}: {counts['ERROR']} errors, {counts['WARNING']} warnings")
        for issue in self.issues[:20]:
            print(f"[{issue['severity']}] {issue['code']}: {issue['message']}")
        if len(self.issues) > 20:
            print(f"... {len(self.issues)-20} additional issues in report.json")
        print(f"Reports: {out}")
        return 1 if counts["ERROR"] else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--state-dir", action="append", default=[], help="State directory relative to project root; repeat for branches")
    p.add_argument("--initial-train", type=int, default=160)
    p.add_argument("--initial-pool", type=int, default=200)
    p.add_argument("--val-size", type=int, default=40)
    p.add_argument("--test-size", type=int, default=40)
    p.add_argument("--shadow-size", type=int, default=100)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--per-cycle", type=int, default=20)
    p.add_argument("--near-duplicates", action="store_true", help="Screen cross-group perceptual matches using dHash")
    p.add_argument("--near-threshold", type=int, default=4)
    p.add_argument("--near-limit", type=int, default=3000)
    p.add_argument("--max-near-pairs", type=int, default=5000, help="Maximum candidate pairs saved to CSV; the scan still counts all matches")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    if min(args.initial_train, args.initial_pool, args.per_cycle) < 0 or args.cycles < 0 or args.max_near_pairs < 1 or args.near_limit < 1 or not 0 <= args.near_threshold <= 64:
        p.error("Invalid nonnegative counts or dHash threshold")
    if args.cycles * args.per_cycle > args.initial_pool:
        p.error("Acquisition budget exceeds the initial pool")
    args.root = args.root.resolve()
    audit = Audit(args.root, args)
    try:
        audit.run()
    except Exception as e:
        audit.issue("ERROR", "AUDIT_EXCEPTION", f"{type(e).__name__}: {e}")
    out = args.output or args.root / "results" / "audits" / ("integrity_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    out = out if out.is_absolute() else args.root / out
    return audit.save(out)


if __name__ == "__main__":
    sys.exit(main())
