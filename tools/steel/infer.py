"""
Inference with a trained checkpoint: boxes in the training CSV format plus a confidence column.

    import sys; sys.path.insert(0, "/path/to/dfine-wrapper")      # not needed when running from the repo root
    from tools.steel.infer import Detector
    det = Detector("outputs/dfine_x_800x128/best.pth")
    df = det.predict("/data/images", conf=0.3)   # folder | image path | list of paths | CSV / DataFrame with image_path

    python tools/steel/infer.py outputs/<name>/best.pth /data/images [more paths] [--conf 0.3] [--out predictions.csv]

predict() returns a pandas DataFrame, one row per box, highest confidence first within an image:
    image_path, instance_label, bbox_x_tl, bbox_y_tl, bbox_x_br, bbox_y_br, split, confidence
Boxes are pixels of the original image, clipped to it. image_path is kept as given (relative paths are made
absolute); split is copied from the input CSV / DataFrame, otherwise the `split` argument. An image without boxes
above `conf` gets one row with empty label / bbox / confidence, as in the training CSV (keep_empty=False drops
those rows), so the output can go straight back into csv_to_coco, e.g. as pseudo-labels.

best.pth written by this repo carries the model config and class names. Older checkpoints and D-FINE's own
last.pth / best_stg*.pth need `config` (configs/_generated/<name>.yml; found automatically for outputs/<name>/).
An ONNX file from tools/steel/export_onnx.py works the same way: Detector("best.onnx") runs it with onnxruntime
(CUDA when onnxruntime-gpu is installed), reading class names and preprocessing from the file's metadata.
"""

import argparse
import contextlib
import io
import json
import math
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from src.core import YAMLConfig  # noqa: E402

COLUMNS = ["image_path", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br", "split", "confidence"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class _Deployed(torch.nn.Module):
    def __init__(self, cfg, deploy=True):
        super().__init__()
        self.model = cfg.model.deploy() if deploy else cfg.model
        self.postprocessor = cfg.postprocessor.deploy()  # (labels, boxes, scores) instead of a list of dicts

    def forward(self, images, orig_sizes):
        return self.postprocessor(self.model(images), orig_sizes)


def _config(ckpt, checkpoint, config):
    if "config" in ckpt:  # embedded by src/solver/det_solver.py; __include__ is already resolved
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.yml"
            p.write_text(yaml.safe_dump({k: v for k, v in ckpt["config"].items() if k != "__include__"}))
            return YAMLConfig(str(p))
    if config is None:
        config = REPO / "configs" / "_generated" / f"{Path(checkpoint).resolve().parent.name}.yml"
        if not config.exists():
            raise FileNotFoundError(f"{checkpoint} has no embedded config: pass config=<the training .yml>")
    return YAMLConfig(str(config))


class _OnnxModel:
    """onnxruntime session called like _Deployed: (images, orig_sizes) -> (labels, boxes, scores) tensors."""

    def __init__(self, path, device=None):
        import onnxruntime as ort

        cuda = device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers()
        providers = (["CUDAExecutionProvider"] if cuda else []) + ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.meta = self.session.get_modelmeta().custom_metadata_map

    def __call__(self, images, orig_sizes):
        out = self.session.run(None, {"images": images.cpu().numpy(),
                                      "orig_target_sizes": orig_sizes.cpu().numpy().astype("int64")})
        return [torch.from_numpy(o) for o in out]


def _class_names(ckpt, ycfg):
    if ckpt.get("class_names"):
        return list(ckpt["class_names"])
    if ycfg.get("remap_mscoco_category") and ycfg.get("num_classes") == 80:  # upstream COCO checkpoints
        from src.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category

        return [mscoco_category2name[mscoco_label2category[i]] for i in range(80)]
    ann = Path(ycfg.get("val_dataloader", {}).get("dataset", {}).get("ann_file", ""))
    if ann.is_file():
        cats = json.loads(ann.read_text())["categories"]
        return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
    return [f"class_{i}" for i in range(ycfg["num_classes"])]


def _abs(p):
    p = str(p)
    return p if os.path.isabs(p) else os.path.abspath(p)


def _collect(images, split):
    """-> (image paths, split per image) from a folder / path / list / CSV / DataFrame."""
    import pandas as pd

    if isinstance(images, (str, Path)) and Path(images).suffix.lower() == ".csv":
        df = pd.read_csv(images)
        base = Path(images).resolve().parent
        df["image_path"] = [p if os.path.isabs(str(p)) else str(base / p) for p in df["image_path"]]
        images = df
    if isinstance(images, pd.DataFrame):
        first = images.drop_duplicates("image_path")
        paths = [_abs(p) for p in first["image_path"]]
        splits = list(first["split"]) if "split" in first else [split] * len(paths)
        return paths, splits
    if isinstance(images, (str, Path)):
        if Path(images).is_dir():
            images = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in IMAGE_EXT)
        else:
            images = [images]
    paths = [_abs(p) for p in images]
    return paths, [split] * len(paths)


class Detector:
    """deploy=True (default, as upstream tools/inference): conv+BN fused, decoder cut at eval_idx - faster, scores
    differ from the training-time validation in the 3rd-4th digit. deploy=False reproduces the validation exactly."""

    def __init__(self, checkpoint, config=None, device=None, deploy=True):
        if Path(checkpoint).suffix.lower() == ".onnx":  # written by tools/steel/export_onnx.py
            self.model, self.config = _OnnxModel(checkpoint, device), None
            self.device = torch.device("cpu")  # inputs go to onnxruntime as numpy arrays
            meta = self.model.meta
            self.class_names = json.loads(meta["class_names"])
            size, norm = json.loads(meta["input_size"]), json.loads(meta["normalize"])
        else:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            cfg = _config(ckpt, checkpoint, config)
            ycfg = self.config = cfg.yaml_cfg
            if isinstance(ycfg.get("HGNetv2"), dict):
                ycfg["HGNetv2"]["pretrained"] = False  # every weight comes from the checkpoint (no download)
            with contextlib.redirect_stdout(io.StringIO()):  # build-time prints
                cfg.model.load_state_dict(ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"])
                model = _Deployed(cfg, deploy)
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = model.to(self.device).eval()
            self.class_names = _class_names(ckpt, ycfg)
            # the val transforms of the training config: Resize to the input size, [0, 1], optional Normalize
            ops = ycfg["val_dataloader"]["dataset"]["transforms"]["ops"]
            size = next((op["size"] for op in ops if op["type"] == "Resize"), ycfg["eval_spatial_size"])
            norm = next(([op["mean"], op["std"]] for op in ops if op["type"] == "Normalize"), None)
        self.input_size = tuple(size)  # (h, w)
        self.normalize = norm  # [mean, std] or None
        self.transform = T.Compose([T.Resize(self.input_size), T.ToTensor()] + ([T.Normalize(*norm)] if norm else []))

    @torch.no_grad()
    def detect(self, paths, conf=0.0, batch_size=16, workers=4):
        """Raw detections per image: yields (path, (w, h), labels, boxes xyxy, scores), best score first."""
        def load(p):
            with Image.open(p) as im:
                im = im.convert("RGB")
            return im, self.transform(im)

        with ThreadPoolExecutor(workers) as pool:
            for i in range(0, len(paths), batch_size):
                chunk = paths[i:i + batch_size]
                loaded = list(pool.map(load, chunk))
                x = torch.stack([t for _, t in loaded]).to(self.device)
                sizes = torch.tensor([[im.width, im.height] for im, _ in loaded], device=self.device)
                labels, boxes, scores = self.model(x, sizes)
                for p, (im, _), lab, box, sc in zip(chunk, loaded, labels, boxes, scores):
                    order = sc.argsort(descending=True)
                    order = order[sc[order] >= conf]
                    yield p, (im.width, im.height), lab[order].cpu(), box[order].cpu(), sc[order].cpu()

    def rows(self, path, size, labels, boxes, scores, split=None, keep_empty=True):
        """One detect() result -> CSV rows (COLUMNS)."""
        w, h = size
        out = []
        for lab, (x0, y0, x1, y1), s in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
            out.append((path, self.class_names[lab], round(min(max(x0, 0), w), 1), round(min(max(y0, 0), h), 1),
                        round(min(max(x1, 0), w), 1), round(min(max(y1, 0), h), 1), split, round(s, 4)))
        if not out and keep_empty:
            out.append((path, None, math.nan, math.nan, math.nan, math.nan, split, math.nan))
        return out

    def predict(self, images, conf=0.3, split="test", batch_size=16, keep_empty=True):
        """Detections in the training CSV format + confidence (see the module docstring)."""
        import pandas as pd

        paths, splits = _collect(images, split)
        rows = []
        for (p, size, lab, box, sc), s in zip(self.detect(paths, conf, batch_size), splits):
            rows += self.rows(p, size, lab, box, sc, s, keep_empty)
        return pd.DataFrame(rows, columns=COLUMNS)


def main():
    ap = argparse.ArgumentParser(description="D-FINE inference -> CSV (training format + confidence)")
    ap.add_argument("checkpoint")
    ap.add_argument("images", nargs="+", help="image files, a folder or a CSV with image_path")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--split", default="test", help="split column value when the input has none")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--config", default=None, help="only for checkpoints without an embedded config")
    ap.add_argument("--no-empty", action="store_true", help="drop the rows of images without boxes")
    ap.add_argument("--out", default=None, help="default: <checkpoint folder>/predictions.csv")
    a = ap.parse_args()
    det = Detector(a.checkpoint, a.config)
    df = det.predict(a.images[0] if len(a.images) == 1 else a.images, a.conf, a.split, a.batch, not a.no_empty)
    out = Path(a.out) if a.out else Path(a.checkpoint).resolve().parent / "predictions.csv"
    df.to_csv(out, index=False)
    n_img = df["image_path"].nunique()
    print(f"{n_img} images, {int(df['instance_label'].notna().sum())} boxes (confidence >= {a.conf}) -> {out}")


if __name__ == "__main__":
    main()
