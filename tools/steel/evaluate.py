"""
Run best.pth of an experiment on its val or test split: predictions, metrics (when the split has boxes), FPS.

    python tools/steel/evaluate.py [experiment.yml] [--split val|test] [--ckpt path] [--conf 0.3] [--fps]

Writes to the run folder:
  <split>_predictions.csv   training CSV columns + confidence (tools/steel/infer.py format), confidence >= conf
  <split>_predictions.json  COCO results (all boxes, used for the metrics)
  eval_<split>.json         mAP@0.5, mAP@0.5:0.95, precision / recall / F1 (best-F1 point, IoU 0.5), AP@0.5 per class
Metrics are skipped for a split without annotations (e.g. an unlabeled test split: inference only).
Metrics use the model exactly as validated during training (Detector(deploy=False)): they match metrics.csv.
--fps: end-to-end speed of the deployed model (conv+BN fused, as tools/steel/infer.py runs it) - batch 1,
per image read + resize + model + postprocess.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig  # noqa: E402
from src.misc.metrics_log import pr_at_best_f1  # noqa: E402
from tools.steel.infer import COLUMNS, Detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", default=str(REPO / "experiment.yml"))
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--ckpt", default=None, help="default: <run>/best.pth")
    ap.add_argument("--conf", type=float, default=0.3, help="confidence threshold for the predictions CSV")
    ap.add_argument("--fps", action="store_true")
    a = ap.parse_args()
    exp = yaml.safe_load(Path(a.experiment).read_text(encoding="utf-8"))
    gen_cfg = REPO / "configs" / "_generated" / f"{exp['name']}.yml"
    ycfg = YAMLConfig(str(gen_cfg)).yaml_cfg
    run_dir = Path(ycfg["output_dir"])
    ckpt = Path(a.ckpt) if a.ckpt else run_dir / "best.pth"
    det = Detector(ckpt, config=gen_cfg, deploy=False)

    if a.split == "val":
        ds = ycfg["val_dataloader"]["dataset"]
        img_dir, ann = ds["img_folder"], Path(ds["ann_file"])
    else:
        if "test_ann" not in ycfg:
            sys.exit("no test split: add split=test rows to the CSV (or data.test_images / test_ann for COCO) "
                     "and run prepare_experiment.py again")
        img_dir, ann = ycfg["test_images"], Path(ycfg["test_ann"])
    coco = json.loads(ann.read_text())
    images = coco["images"]
    paths = [os.path.join(img_dir, im["file_name"]) for im in images]  # absolute file_name wins

    res, rows = [], []
    for im, (p, size, lab, box, sc) in zip(images, det.detect(paths)):
        for (x1, y1, x2, y2), c, s in zip(box.tolist(), lab.tolist(), sc.tolist()):
            res.append({"image_id": im["id"], "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(s)})
        keep = sc >= a.conf
        rows += det.rows(p, size, lab[keep], box[keep], sc[keep], a.split)
    (run_dir / f"{a.split}_predictions.json").write_text(json.dumps(res))
    csv_path = run_dir / f"{a.split}_predictions.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, index=False)

    out = {"checkpoint": str(ckpt), "split": a.split, "images": len(images)}
    if coco["annotations"]:
        from faster_coco_eval import COCO, COCOeval_faster

        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(ann))
            e = COCOeval_faster(gt, gt.loadRes(res), iouType="bbox", print_function=print, separate_eval=True)
            e.evaluate()
            e.accumulate()
            e.summarize()
        p, r, f1, ap50 = pr_at_best_f1(e)
        cls_names = [c["name"] for c in gt.loadCats(sorted(gt.getCatIds()))]
        out.update({"mAP50": float(e.stats[1]), "mAP50_95": float(e.stats[0]), "precision": p, "recall": r,
                    "f1": f1, "AP50_per_class": dict(zip(cls_names, ap50))})

    if a.fps:
        del det
        fast = Detector(ckpt, config=gen_cfg)
        for _ in fast.detect(paths[:20], batch_size=1):  # warm-up
            pass
        n = min(300, len(paths))
        t0 = time.perf_counter()
        for _ in fast.detect(paths[:n], batch_size=1, workers=1):  # boxes are copied to the CPU: synchronised
            pass
        out["fps"] = n / (time.perf_counter() - t0)

    (run_dir / f"eval_{a.split}.json").write_text(json.dumps(out, indent=1))
    print(f"checkpoint {ckpt} | split {a.split}: {len(images)} images, {len(res)} boxes "
          f"-> {csv_path} (confidence >= {a.conf})")
    if "mAP50" in out:
        print(f"mAP@0.5 {out['mAP50']:.4f} | mAP@0.5:0.95 {out['mAP50_95']:.4f} | "
              f"P {out['precision']:.3f} R {out['recall']:.3f} F1 {out['f1']:.3f}")
        print("AP@0.5 per class: " + ", ".join(f"{k} {v:.3f}" for k, v in out["AP50_per_class"].items()))
    else:
        print("no annotations in this split: predictions only")
    if a.fps:
        print(f"FPS (batch 1, FP32, read+resize+model+postprocess): {out['fps']:.1f}")


if __name__ == "__main__":
    main()
