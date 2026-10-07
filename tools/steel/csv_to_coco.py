"""
Convert a box CSV to COCO json, one file per split (train / val / test).

Columns (one row per box; extra columns are ignored):
    image_path       absolute path to the image
    instance_label   class name (string)
    bbox_x_tl, bbox_y_tl, bbox_x_br, bbox_y_br   top-left / bottom-right corner, pixels of the original image
    split            train | val | test   (test is kept for inference; it may have boxes or not)
A row with empty label / bbox registers the image without boxes (e.g. defect-free or unlabeled test images).
The legacy steel CSV (ImageId, ClassId, x_min, y_min, x_max, y_max, split + an images folder) is detected by its header.

Images are not copied: file_name is the absolute image path (D-FINE / torchvision join it with img_folder, and an
absolute file_name wins). Labels are optionally merged (class_merge: {old: new}) and mapped to contiguous category
ids 0..K-1 in sorted order (numeric-aware), the same for every split.

python tools/steel/csv_to_coco.py data.csv out_dir [--images DIR] [--merge old:new,old:new]
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image

NEW = {"image": "image_path", "label": "instance_label", "x0": "bbox_x_tl", "y0": "bbox_y_tl", "x1": "bbox_x_br",
       "y1": "bbox_y_br", "split": "split"}
LEGACY = {"image": "ImageId", "label": "ClassId", "x0": "x_min", "y0": "y_min", "x1": "x_max", "y1": "y_max",
          "split": "split"}
SPLITS = ("train", "val", "test")


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _label_key(label):  # numeric-aware sort: "2" < "10"
    return (0, float(label), label) if _num(label) is not None else (1, 0.0, label)


def convert(csv_path, out_dir, images_dir=None, class_merge=None):
    csv_path = Path(csv_path)
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8-sig")))
    header = set(rows[0]) if rows else set()
    if set(NEW.values()) <= header:
        col, legacy = NEW, False
    elif set(LEGACY.values()) <= header:
        col, legacy = LEGACY, True
        assert images_dir, "legacy CSV (ImageId, ...) needs the images folder"
    else:
        raise ValueError(f"{csv_path}: expected columns {sorted(NEW.values())} (or the legacy steel columns)")
    merge = {str(k): str(v) for k, v in (class_merge or {}).items()}
    base = Path(images_dir) if images_dir else csv_path.parent

    def label_of(r):
        raw = (r[col["label"]] or "").strip()
        if not raw or raw.lower() == "nan":
            return None
        raw = str(int(float(raw))) if legacy and _num(raw) is not None else raw
        return merge.get(raw, raw)

    def path_of(r):
        p = Path(r[col["image"]].strip())
        return (p if p.is_absolute() else base / p).as_posix()

    labels = sorted({lb for r in rows if (lb := label_of(r)) is not None}, key=_label_key)
    cat_id = {lb: i for i, lb in enumerate(labels)}
    categories = [{"id": cat_id[lb], "name": f"defect_{lb}" if legacy else lb, "supercategory": "object"}
                  for lb in labels]

    by_split, skipped = defaultdict(lambda: defaultdict(list)), defaultdict(int)
    for r in rows:
        s = (r[col["split"]] or "").strip().lower()
        if s not in SPLITS:
            skipped[s] += 1
            continue
        by_split[s][path_of(r)].append(r)
    for s, n in skipped.items():
        print(f"WARNING: {n} rows with split={s!r} ignored (expected {', '.join(SPLITS)})")
    seen = defaultdict(set)
    for s, imgs in by_split.items():
        for p in imgs:
            seen[p].add(s)
    shared = [p for p, ss in seen.items() if len(ss) > 1]
    if shared:
        print(f"WARNING: {len(shared)} images are in more than one split (e.g. {shared[0]}) - check for leakage")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split in SPLITS:
        if split not in by_split:
            continue
        images, anns = [], []
        for i, (path, boxes) in enumerate(sorted(by_split[split].items()), start=1):
            with Image.open(path) as im:
                w, h = im.size
            images.append({"id": i, "file_name": path, "width": w, "height": h})
            for r in boxes:
                lb = label_of(r)
                x0, y0, x1, y1 = (_num(r[col[k]]) for k in ("x0", "y0", "x1", "y1"))
                if lb is None or None in (x0, y0, x1, y1):
                    continue  # image without boxes
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cat_id[lb],
                             "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0})
        path = out_dir / f"{split}.json"
        path.write_text(json.dumps({"images": images, "annotations": anns, "categories": categories}))
        written[split] = path
        print(f"{split}: {len(images)} images, {len(anns)} boxes -> {path}")
    print(f"classes ({len(labels)}): " + ", ".join(f"{c['id']}={c['name']}" for c in categories))
    return written, categories


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out_dir")
    ap.add_argument("--images", default=None, help="folder for relative / legacy ImageId paths")
    ap.add_argument("--merge", default="", help="old:new label pairs, e.g. scratch_a:scratch,6:1")
    a = ap.parse_args()
    convert(a.csv, a.out_dir, a.images, dict(p.split(":", 1) for p in a.merge.split(",") if p))
