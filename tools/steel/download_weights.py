"""
Download the weights an experiment needs (GitHub releases of the D-FINE authors).

    python tools/steel/download_weights.py                       # model + pretrain of experiment.yml
    python tools/steel/download_weights.py x                     # a given size: n | s | m | l | x | all
    python tools/steel/download_weights.py x --pretrain coco     # obj365 | obj2coco | coco | all
    python tools/steel/download_weights.py x --backbone          # + the HGNetv2 backbone (only for weights: none)

  detector  weights/dfine_<m>_<pretrain>.pth   start point of the fine-tuning (train.py -t)
            obj365    Objects365 only (s, m, l, x) - the authors' recommendation for custom datasets
            obj2coco  Objects365, then COCO (s, m, l, x; best COCO AP)
            coco      COCO only (n, s, m, l, x)
  backbone  weight/hgnetv2/PPHGNetV2_<B>_stage1.pth   ImageNet HGNetv2 - read only when training starts from the
            bare backbone (experiment.yml: weights: none); D-FINE would download it itself from the same place.

Existing files are kept. A Git LFS pointer in place of a file is fetched with `git lfs pull` first.
Restricted networks: a proxy via HTTPS_PROXY / HTTP_PROXY, or copy the files from another machine into weights/
and weight/hgnetv2/.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASE = "https://github.com/Peterande/storage/releases/download/dfinev1.0/"
MODELS = ("n", "s", "m", "l", "x")
CHECKPOINTS = {  # pretrain -> model -> file name (README model zoo)
    "obj365": {m: f"dfine_{m}_obj365.pth" for m in "smlx"},
    "obj2coco": {"s": "dfine_s_obj2coco.pth", "m": "dfine_m_obj2coco.pth", "l": "dfine_l_obj2coco_e25.pth",
                 "x": "dfine_x_obj2coco.pth"},
    "coco": {m: f"dfine_{m}_coco.pth" for m in MODELS},
}
BACKBONE = {"n": "B0", "s": "B0", "m": "B2", "l": "B4", "x": "B5"}  # HGNetv2.name in configs/dfine/*


def checkpoint_path(model, pretrain):
    if model not in CHECKPOINTS[pretrain]:
        raise ValueError(f"D-FINE-{model.upper()} has no {pretrain} checkpoint; "
                         f"available: {', '.join(p for p in CHECKPOINTS if model in CHECKPOINTS[p])}")
    return REPO / "weights" / CHECKPOINTS[pretrain][model]


def backbone_path(model):
    return REPO / "weight" / "hgnetv2" / f"PPHGNetV2_{BACKBONE[model]}_stage1.pth"


def is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(64).startswith(b"version https://git-lfs")


def exists(path):
    if not (path.exists() and path.stat().st_size > 0):
        return False
    if is_lfs_pointer(path):  # repo cloned without git-lfs content
        rel = path.relative_to(REPO).as_posix()
        print(f"{rel} is a Git LFS pointer -> git lfs pull --include {rel}")
        r = subprocess.run(["git", "lfs", "pull", "--include", rel], cwd=REPO)
        if r.returncode != 0 or is_lfs_pointer(path):
            print("git lfs pull failed (git-lfs missing or the LFS host unreachable); downloading instead")
            return False
    print(f"ok (exists)  {path.relative_to(REPO).as_posix()}")
    return True


def download(path):
    import torch

    if exists(path):
        return
    url = RELEASE + path.name
    print(f"downloading {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.hub.download_url_to_file(url, str(tmp), progress=True)
        torch.load(tmp, map_location="cpu", weights_only=True)  # a truncated / HTML file fails here
        os.replace(tmp, path)
    except Exception as e:
        sys.exit(f"download failed: {type(e).__name__}: {str(e)[:300]}\n{url} may be blocked on this network: "
                 "set HTTPS_PROXY, or copy the file from another machine to "
                 f"{path.relative_to(REPO).as_posix()}")
    finally:
        tmp.unlink(missing_ok=True)
    print(f"ok           {path.relative_to(REPO).as_posix()} ({path.stat().st_size / 2**20:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="*", help="n s m l x | all (default: model of experiment.yml)")
    ap.add_argument("--pretrain", default=None, help="obj365 | obj2coco | coco | all (default: experiment.yml)")
    ap.add_argument("--backbone", action="store_true", help="also the HGNetv2 backbone (weights: none)")
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--experiment", default=str(REPO / "experiment.yml"))
    a = ap.parse_args()

    exp = {}
    if Path(a.experiment).exists():
        import yaml

        exp = yaml.safe_load(Path(a.experiment).read_text(encoding="utf-8")) or {}
    models = a.models or [str(exp.get("model", "x")).lower()]
    models = list(MODELS) if "all" in models else models
    pretrain = (a.pretrain or str(exp.get("pretrain", "obj365"))).lower()
    if pretrain not in (*CHECKPOINTS, "all"):
        sys.exit(f"pretrain must be one of {', '.join(CHECKPOINTS)} or all, not {pretrain!r}")
    pretrains = list(CHECKPOINTS) if pretrain == "all" else [pretrain]
    for m in models:
        if m not in MODELS:
            sys.exit(f"model must be one of {', '.join(MODELS)} or all, not {m!r}")
        if not a.no_detector:
            for p in pretrains:
                if m in CHECKPOINTS[p]:
                    download(checkpoint_path(m, p))
                elif len(pretrains) == 1 and len(models) == 1:
                    sys.exit(f"D-FINE-{m.upper()} has no {p} checkpoint; available: "
                             + ", ".join(q for q in CHECKPOINTS if m in CHECKPOINTS[q]))
                elif len(pretrains) == 1:
                    print(f"skip         D-FINE-{m.upper()} has no {p} checkpoint")
        if a.backbone:
            download(backbone_path(m))


if __name__ == "__main__":
    main()
