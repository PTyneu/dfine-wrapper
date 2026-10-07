"""
Export a trained checkpoint to ONNX and check it against PyTorch.

    python tools/steel/export_onnx.py                                # best.pth of experiment.yml -> best.onnx next to it
    python tools/steel/export_onnx.py my_experiment.yml
    python tools/steel/export_onnx.py path/to/best.pth [--out model.onnx] [--images a.jpg b.jpg]
    python tools/steel/export_onnx.py weights/dfine_x_coco.pth --config configs/dfine/dfine_hgnetv2_x_coco.yml

The model config and class names come from best.pth (training stores them there), so the export matches the trained
model even if experiment.yml was edited since: experiment.yml only locates <output_root>/<name>/best.pth.
Checkpoints without an embedded config (the upstream D-FINE weights, older best.pth) need --config.

ONNX graph: batch N is dynamic, the input size is fixed to the training input_size [H, W]
    images             float32 [N, 3, H, W]  RGB resized to W x H (bilinear), /255, then (x - mean) / std
    orig_target_sizes  int64   [N, 2]        original (width, height)
    labels int64 [N, 300], boxes float32 [N, 300, 4] (x_tl, y_tl, x_br, y_br in original pixels), scores float32 [N, 300]
Postprocessing (sigmoid, top-300) is inside the graph, no NMS needed. Metadata (custom_metadata_map):
class_names (JSON list, index = label), input_size [H, W], normalize [mean, std] or null.
Run it with Detector("best.onnx") from tools/steel/infer.py, or with plain onnxruntime (README).

The check runs PyTorch and onnxruntime on CPU over a few images (--images, else the first val images of the
training config) in batches of 3 and 1. Bit-exact equality is not expected: CPU kernels differ in the last bits and
the decoder's top-k query selection can swap near-tied queries (most on weakly trained models; a 1-epoch steel
model: sorted scores within 9e-3, val AP50 0.1756 vs 0.1740 in PyTorch). It passes when the sorted scores differ by
< 0.02 and >= 90% of the ONNX top-20 detections are also found by PyTorch (same class, IoU >= 0.9, score within
0.01); a broken export (wrong size, preprocessing, branch) misses both by far.
"""

import argparse
import inspect
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.steel.infer import Detector  # noqa: E402
from tools.steel.prepare_experiment import path_of  # noqa: E402


def export(checkpoint, out=None, config=None, opset=17):
    """checkpoint (.pth) -> ONNX file with metadata; returns (the PyTorch Detector, the ONNX path)."""
    import onnx

    det = Detector(checkpoint, config, device="cpu")
    h, w = det.input_size
    # batch 2: the decoder's `if batch > 1: anchors.repeat(batch, ...)` branch is traced, correct for any N
    x, sizes = torch.rand(2, 3, h, w), torch.tensor([[w, h], [w, h]])
    out = Path(out) if out else Path(checkpoint).with_suffix(".onnx")
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # TracerWarnings for shape-dependent Python branches (sizes are fixed)
        torch.onnx.export(det.model, (x, sizes), str(out), input_names=["images", "orig_target_sizes"],
                          output_names=["labels", "boxes", "scores"], opset_version=opset, do_constant_folding=True,
                          dynamic_axes={n: {0: "N"} for n in ("images", "orig_target_sizes", "labels", "boxes",
                                                              "scores")}, **extra)
    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    meta = {"class_names": json.dumps(det.class_names), "input_size": json.dumps(list(det.input_size)),
            "normalize": json.dumps(det.normalize), "source": Path(checkpoint).name}
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(model, str(out))
    return det, out


def check(det, onnx_path, paths):
    """-> (max |diff| of the sorted scores, share of the ONNX top-20 detections PyTorch also has)."""
    from torchvision.ops import box_iou

    ort = Detector(onnx_path, device="cpu")
    score_diff, found, total = 0.0, 0, 0
    for (_, _, tl, tb, ts), (_, _, ol, ob, os_) in zip(det.detect(paths, batch_size=3),
                                                        ort.detect(paths, batch_size=3)):
        score_diff = max(score_diff, (ts - os_).abs().max().item())
        ol, ob, os_ = ol[:20], ob[:20], os_[:20]
        same = (box_iou(ob, tb) >= 0.9) & (ol[:, None] == tl[None]) & ((os_[:, None] - ts[None]).abs() <= 0.01)
        found += int(same.any(1).sum())
        total += len(ol)
    return score_diff, found / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description="D-FINE checkpoint -> ONNX (+ check against PyTorch)")
    ap.add_argument("target", nargs="?", default=str(REPO / "experiment.yml"), help="experiment.yml or a .pth")
    ap.add_argument("--config", default=None, help="only for checkpoints without an embedded config")
    ap.add_argument("--out", default=None, help="default: the checkpoint path with .onnx")
    ap.add_argument("--images", nargs="*", default=None, help="images for the check (default: first val images)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()

    target = Path(a.target)
    if target.suffix.lower() in (".yml", ".yaml"):
        exp = yaml.safe_load(target.read_text(encoding="utf-8"))
        ckpt = path_of(exp.get("output_root", "outputs"), target.resolve().parent) / exp["name"] / "best.pth"
        if not ckpt.exists():
            sys.exit(f"{ckpt} not found: train first (bash train.sh {a.target}) or pass a .pth")
    else:
        ckpt = target
    det, out = export(ckpt, a.out, a.config, a.opset)
    print(f"{out}  ({out.stat().st_size / 2**20:.0f} MB) | input {det.input_size[0]}x{det.input_size[1]}, "
          f"batch dynamic | {len(det.class_names)} classes: {', '.join(det.class_names[:8])}"
          + (" ..." if len(det.class_names) > 8 else ""))
    if a.no_check:
        return
    paths = a.images
    if not paths and det.config:
        ds = det.config["val_dataloader"]["dataset"]
        if Path(ds["ann_file"]).is_file():
            images = json.loads(Path(ds["ann_file"]).read_text())["images"][:4]
            paths = [os.path.join(ds["img_folder"], im["file_name"]) for im in images]
    if not paths:
        print("check skipped: pass --images")
        return
    score_diff, found = check(det, out, paths)
    ok = score_diff < 0.02 and found >= 0.9
    print(f"check on {len(paths)} images, onnxruntime vs PyTorch: max |score diff| {score_diff:.1e}, "
          f"top-20 detections found by both {found:.0%} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
