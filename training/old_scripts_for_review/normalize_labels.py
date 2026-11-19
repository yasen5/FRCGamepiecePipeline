#!/usr/bin/env python3
"""Normalize class labels across all downloaded Charged Up (2023 FRC) datasets
to a single canonical label set, remapping YOLO class indices in every
label .txt file and rewriting each dataset's data.yaml accordingly.
"""
import re
from pathlib import Path
import yaml

RAW_DIR = Path("/mnt/ssd2/data/gaussian_splatting/robotics/charged_up/raw_datasets")

# Canonical game-piece classes for 2023 Charged Up.
CANONICAL = ["cone", "cube"]

# Maps every raw class name variant (lowercased, whitespace-collapsed) seen
# across the 5 datasets to a canonical class. "tipped_cone" is a cone that
# has fallen over -- same game piece, different orientation -- so it folds
# into "cone" rather than becoming a third class.
SYNONYMS = {
    "cone": "cone",
    "cones": "cone",
    "yellow cone": "cone",
    "yellow_cone": "cone",
    "tipped_cone": "cone",
    "tipped cone": "cone",
    "fallen_cone": "cone",
    "cube": "cube",
    "cubes": "cube",
    "purple cube": "cube",
    "purple_cube": "cube",
    "box": "cube",
}


def normalize_name(name: str) -> str:
    key = re.sub(r"\s+", " ", name.strip().lower())
    if key not in SYNONYMS:
        raise ValueError(f"Unmapped class name: {name!r} (add it to SYNONYMS)")
    return SYNONYMS[key]


def process_dataset(ds_dir: Path):
    yaml_path = ds_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"  SKIP {ds_dir.name}: no data.yaml")
        return

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    old_names = data["names"]
    # old class index -> canonical class index
    remap = {}
    for old_idx, old_name in enumerate(old_names):
        canon = normalize_name(old_name)
        remap[old_idx] = CANONICAL.index(canon)

    print(f"  {ds_dir.name}: {old_names} -> remap {remap}")

    if old_names == CANONICAL and all(i == remap[i] for i in remap):
        print(f"    already canonical, only rewriting names/nc for consistency")

    label_files = list(ds_dir.rglob("labels/*.txt"))
    changed = 0
    class_counts = {c: 0 for c in CANONICAL}
    for lf in label_files:
        lines = lf.read_text().splitlines()
        new_lines = []
        dirty = False
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            old_cls = int(parts[0])
            new_cls = remap[old_cls]
            class_counts[CANONICAL[new_cls]] += 1
            if new_cls != old_cls:
                dirty = True
            parts[0] = str(new_cls)
            new_lines.append(" ".join(parts))
        if dirty:
            lf.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            changed += 1

    data["names"] = CANONICAL
    data["nc"] = len(CANONICAL)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"    {len(label_files)} label files, {changed} rewritten")
    print(f"    class counts: {class_counts}")


def main():
    for ds_dir in sorted(RAW_DIR.iterdir()):
        if ds_dir.is_dir():
            process_dataset(ds_dir)


if __name__ == "__main__":
    main()
