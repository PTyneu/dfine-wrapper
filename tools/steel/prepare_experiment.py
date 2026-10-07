"""
Build a complete D-FINE training config from the single editable file (experiment.yml) and optionally train.

    python tools/steel/prepare_experiment.py experiment.yml           # -> configs/_generated/<name>.yml + command
    python tools/steel/prepare_experiment.py experiment.yml --train   # ... and run train.py with it
                                                                     #     (several `devices`: DDP via torchrun)

What is derived from experiment.yml (everything else comes from the upstream fine-tuning recipe of the model,
which stays untouched - the upstream way of training still works):
  * recipe: configs/dfine/custom/objects365/dfine_hgnetv2_<m>_obj2custom.yml for Objects365 weights,
    configs/dfine/custom/dfine_hgnetv2_<m>_custom.yml otherwise (COCO / Objects365+COCO weights, bare backbone)
  * data: CSV (converted to COCO json once, cached in data_cache/) or ready COCO json + image folders;
    category ids are remapped to 0..K-1 when needed; num_classes is read from the annotations
  * input size [h, w]: Resize in train/val, eval_spatial_size; rectangular inputs drop the multi-scale batches
    (they build square images)
  * the recipe's schedule scaled to `epochs`: augmentations and multi-scale stop for the same share of the last
    epochs as in the recipe, where D-FINE also reloads the best stage-1 weights and restarts the EMA
  * warm-ups: lr 0.5 epoch (0 when the recipe has none, as obj2custom); EMA 1.5 epochs always (like deim-steel):
    obj2custom's `ema.warmups: 0` keeps the EMA decay at 0.9999 from the first step, so after the few thousand steps
    of a small fine-tuning the EMA weights - the ones evaluated and saved - are still mostly the start checkpoint
  * optimizer: AdamW (recipe lr scaled linearly to the batch) or SGD (lr 0.01, the recipe's backbone ratio,
    Nesterov, clip 10)
  * run folder outputs/<name>/: best.pth only + metrics.csv / metrics.png / log.txt (see src/solver/det_solver.py)
  * devices: GPU ids; more than one -> DDP on one machine (torchrun). batch_size stays the total over all GPUs
    (D-FINE's total_batch_size), so iterations per epoch, lr and warmups do not depend on the number of GPUs
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.core.yaml_utils import load_config  # noqa: E402
from tools.steel.download_weights import MODELS, backbone_path, checkpoint_path, is_lfs_pointer  # noqa: E402


def path_of(p, base):
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return (p if p.is_absolute() else (base / p)).resolve()


def auto(v):
    return v is None or (isinstance(v, str) and v.lower() == "auto")


def gpu_ids(v):
    """devices: 0 | [0, 1] | "0,1" -> ['0', '1']; absent -> None (CUDA_VISIBLE_DEVICES is left as it is, one GPU)."""
    if v is None:
        return None
    ids = [str(int(x)) for x in v] if isinstance(v, (list, tuple)) else [s.strip() for s in str(v).split(",")]
    return [i for i in ids if i] or None


def contiguous_ann(ann, cache_dir):
    """D-FINE (remap_mscoco_category: False) needs category ids 0..K-1; write a remapped copy otherwise."""
    d = json.loads(Path(ann).read_text())
    ids = sorted(c["id"] for c in d["categories"])
    if ids == list(range(len(ids))):
        return Path(ann), len(ids), len(d["images"])
    remap = {old: new for new, old in enumerate(ids)}
    for c in d["categories"]:
        c["id"] = remap[c["id"]]
    for a in d["annotations"]:
        a["category_id"] = remap[a["category_id"]]
    out = cache_dir / (Path(ann).stem + "_ids0.json")
    out.write_text(json.dumps(d))
    return out, len(ids), len(d["images"])


def prepare_data(data, base, name):
    cache = REPO / "data_cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    test = None
    if data.get("format", "coco") == "csv":
        from tools.steel.csv_to_coco import convert

        csv_path = path_of(data["csv"], base)
        if not csv_path.is_file():
            sys.exit(f"data.csv not found: {csv_path} (relative paths are relative to the experiment file)")
        images = path_of(data["images"], base) if data.get("images") else None  # legacy ImageId / relative paths
        merge = {str(k): str(v) for k, v in (data.get("class_merge") or {}).items()}
        key = hashlib.md5(f"v2|{csv_path}|{os.path.getmtime(csv_path)}|{images}|{sorted(merge.items())}".encode())
        stamp = cache / f"source_{key.hexdigest()[:10]}.txt"
        if not stamp.exists():
            for old in list(cache.glob("*.json")) + list(cache.glob("source_*.txt")):
                old.unlink()  # stale splits of an earlier conversion
            convert(csv_path, cache, images, merge)
            stamp.write_text(f"{csv_path}\n{images}\n{merge}\n")
        root = images or csv_path.parent  # file_name is an absolute path, it wins over the folder
        train_img = val_img = root
        train_ann, val_ann = cache / "train.json", cache / "val.json"
        if (cache / "test.json").exists():
            test = (root, cache / "test.json")
    else:
        train_img, val_img = path_of(data["train_images"], base), path_of(data["val_images"], base)
        train_ann, val_ann = path_of(data["train_ann"], base), path_of(data["val_ann"], base)
        if data.get("test_ann"):
            test = (path_of(data["test_images"], base), path_of(data["test_ann"], base))
    train_ann, k_train, n_train = contiguous_ann(train_ann, cache)
    val_ann, k_val, _ = contiguous_ann(val_ann, cache)
    assert k_train == k_val, f"train has {k_train} classes, val has {k_val}"
    if test:
        test = (test[0], contiguous_ann(test[1], cache)[0])
    return train_img, train_ann, val_img, val_ann, k_train, n_train, test


def recipe_file(model, pretrain, from_backbone):
    if pretrain == "obj365" and not from_backbone:
        return REPO / "configs" / "dfine" / "custom" / "objects365" / f"dfine_hgnetv2_{model}_obj2custom.yml"
    return REPO / "configs" / "dfine" / "custom" / f"dfine_hgnetv2_{model}_custom.yml"


def build(exp, base):
    name, model = exp["name"], str(exp.get("model", "x")).lower()
    pretrain = str(exp.get("pretrain", "obj365")).lower()
    assert model in MODELS, f"model must be one of {MODELS}"
    assert pretrain in ("obj365", "obj2coco", "coco"), "pretrain must be obj365, obj2coco or coco"

    weights = exp.get("weights", "auto")
    if str(weights).lower() == "none":  # bare backbone: recipe `custom` whatever `pretrain` says
        weights = None
    else:  # checkpoint_path also rejects a model / pretrain pair without checkpoint and recipe (e.g. n + obj365)
        weights = checkpoint_path(model, pretrain) if auto(weights) else path_of(weights, base)
    rfile = recipe_file(model, pretrain, weights is None)
    recipe = load_config(str(rfile))
    r_epochs, r_stop = recipe["epochs"], recipe["train_dataloader"]["collate_fn"]["stop_epoch"]
    r_batch = recipe["train_dataloader"]["total_batch_size"]
    r_collate = recipe["train_dataloader"]["collate_fn"]

    train_img, train_ann, val_img, val_ann, num_classes, n_train, test = prepare_data(exp["data"], base, name)
    E, batch, val_batch = int(exp.get("epochs", 12)), int(exp.get("batch_size", 16)), int(exp.get("val_batch_size", 16))
    n_gpu = len(gpu_ids(exp.get("devices")) or [0])
    if batch % n_gpu or val_batch % n_gpu:
        raise ValueError(f"batch_size ({batch}) and val_batch_size ({val_batch}) are totals over all GPUs: "
                         f"make them divisible by the number of devices ({n_gpu})")
    h, w = (int(v) for v in exp.get("input_size", [640, 640]))
    assert h % 32 == 0 and w % 32 == 0, "input_size sides must be multiples of 32"
    multiscale = bool(exp.get("multiscale", False))
    if str(exp.get("best_metric", "map50")) not in ("map50", "map"):
        raise ValueError("best_metric must be map50 (AP@0.5) or map (AP@0.5:0.95)")
    if multiscale and h != w:
        raise ValueError("multi-scale batches are square; use them only with a square input_size")

    # schedule: the clean tail (no augmentations / multi-scale, EMA restarted from the best stage-1 weights) keeps
    # the recipe's share; runs shorter than 3 epochs have none (stop == E is never reached)
    tail = max(1, round(E * (r_epochs - r_stop) / r_epochs)) if E >= 3 else 0
    stop = E - tail
    iters = max(1, n_train // batch)
    warmup = max(50, round(0.5 * iters)) if recipe["lr_warmup_scheduler"]["warmup_duration"] else 0
    ema_warmups = max(100, round(1.5 * iters))  # never 0: see the module docstring

    # optimizer: the recipe's parameter groups; the backbone group carries its own lr
    opt = str(exp.get("optimizer", "adamw")).lower()
    groups = [dict(g) for g in recipe["optimizer"]["params"]]
    r_lr = recipe["optimizer"]["lr"]
    r_ratio = next(g["lr"] for g in groups if "lr" in g) / r_lr
    ratio = r_ratio if auto(exp.get("backbone_lr_ratio")) else float(exp["backbone_lr_ratio"])
    if opt == "sgd":
        lr = 0.01 if auto(exp.get("lr")) else float(exp["lr"])
        wd = 1e-4 if auto(exp.get("weight_decay")) else float(exp["weight_decay"])
    elif opt == "adamw":
        lr = r_lr * batch / r_batch if auto(exp.get("lr")) else float(exp["lr"])
        wd = recipe["optimizer"]["weight_decay"] if auto(exp.get("weight_decay")) else float(exp["weight_decay"])
    else:
        raise ValueError("optimizer must be adamw or sgd")
    for g in groups:
        if "lr" in g:  # the backbone group
            g["lr"] = lr * ratio
    optimizer = {"type": "AdamW", "params": groups, "lr": lr, "betas": [0.9, 0.999], "weight_decay": wd}
    if opt == "sgd":  # SGDIgnoreBetas tolerates the AdamW `betas` key merged in from the base configs
        optimizer = {"type": "SGDIgnoreBetas", "params": groups, "lr": lr, "momentum": 0.9, "nesterov": True,
                     "weight_decay": wd}

    # transforms: recipe ops with the requested size
    tf = recipe["train_dataloader"]["dataset"]["transforms"]
    train_ops = [dict(op, size=[h, w]) if op["type"] == "Resize" else dict(op) for op in tf["ops"]]
    val_ops = [dict(op, size=[h, w]) if op["type"] == "Resize" else dict(op)
               for op in recipe["val_dataloader"]["dataset"]["transforms"]["ops"]]

    run_dir = path_of(exp.get("output_root", "outputs"), base) / name
    cfg = {
        "__include__": [Path(os.path.relpath(rfile, REPO / "configs" / "_generated")).as_posix()],
        "output_dir": run_dir.as_posix(),
        "num_classes": num_classes,
        "remap_mscoco_category": False,
        "eval_spatial_size": [h, w],
        "epochs": E,
        "print_freq": 100,
        "checkpoint_freq": 10 ** 6,
        "save_best_only": bool(exp.get("save_best_only", True)),
        "best_metric": str(exp.get("best_metric", "map50")),
        "plot_metrics": True,
        # bare-backbone start: ImageNet HGNetv2 from weight/hgnetv2/ (train.py turns this off for -t)
        "HGNetv2": {"pretrained": weights is None, "local_model_dir": (REPO / "weight" / "hgnetv2").as_posix() + "/"},
        "lr_warmup_scheduler": {"warmup_duration": warmup},
        "ema": {"warmups": ema_warmups},
        "optimizer": optimizer,
        "train_dataloader": {
            "total_batch_size": batch,
            "num_workers": int(exp.get("workers", 4)),
            "dataset": {"img_folder": Path(train_img).as_posix(), "ann_file": Path(train_ann).as_posix(),
                        "transforms": {"ops": train_ops, "policy": dict(tf["policy"], epoch=stop)}},
            "collate_fn": {"stop_epoch": stop, "ema_restart_decay": r_collate.get("ema_restart_decay", 0.9999),
                           "base_size": h, "base_size_repeat": r_collate.get("base_size_repeat") if multiscale else None},
        },
        "val_dataloader": {
            "total_batch_size": val_batch,
            "num_workers": int(exp.get("workers", 4)),
            "dataset": {"img_folder": Path(val_img).as_posix(), "ann_file": Path(val_ann).as_posix(),
                        "transforms": {"ops": val_ops}},
        },
    }
    if opt == "sgd":
        cfg["clip_max_norm"] = 10.0  # 0.1 of the AdamW recipe would stall plain SGD
    if test:  # not used by training; tools/steel/evaluate.py --split test
        cfg["test_images"], cfg["test_ann"] = Path(test[0]).as_posix(), Path(test[1]).as_posix()
    summary = (f"model D-FINE-{model.upper()} ({'backbone only' if weights is None else pretrain} start, recipe "
               f"{rfile.stem}) | input {h}x{w} | {E} epochs, batch {batch}, {opt} lr {lr:.3g} (backbone x{ratio:.3g}), "
               f"wd {wd:.3g} | warmup {warmup} it, EMA warmup {ema_warmups} it | classes {num_classes}, train images "
               f"{n_train} | aug stop {stop}, multiscale {multiscale} | test split {bool(test)}"
               + (f" | DDP on {n_gpu} GPUs, batch {batch // n_gpu} per GPU" if n_gpu > 1 else ""))
    return cfg, weights, run_dir, summary, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", default=str(REPO / "experiment.yml"))
    ap.add_argument("--train", action="store_true", help="start train.py after writing the config")
    args = ap.parse_args()
    exp_path = Path(args.experiment).resolve()
    exp = yaml.safe_load(exp_path.read_text(encoding="utf-8"))
    cfg, weights, run_dir, summary, model = build(exp, exp_path.parent)

    out = REPO / "configs" / "_generated" / f"{exp['name']}.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# generated by tools/steel/prepare_experiment.py from {exp_path.name} - edit that file instead\n"
                   + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    pretrain = str(exp.get("pretrain", "obj365")).lower()
    if weights is not None and not Path(weights).exists():
        sys.exit(f"weights not found: {weights}\nrun: bash scripts/download_weights.sh {model} --pretrain {pretrain}")
    if weights is not None and is_lfs_pointer(weights):
        sys.exit(f"{weights} is a Git LFS pointer, not the weights.\n"
                 "run: git lfs install && git lfs pull   (or: bash scripts/download_weights.sh)")
    if weights is None and not backbone_path(model).exists():  # HGNetv2 would try to download it at start
        sys.exit(f"backbone weights not found: {backbone_path(model)}\n"
                 f"run: bash scripts/download_weights.sh {model} --backbone --no-detector")

    train = ["train.py", "-c", out.relative_to(REPO).as_posix(), "--seed", str(exp.get("seed", 0))]
    if exp.get("amp", True):
        train.append("--use-amp")
    if weights is not None:
        train += ["-t", Path(weights).as_posix()]
    ids = gpu_ids(exp.get("devices"))
    ddp = bool(ids) and len(ids) > 1
    if ddp:  # one process per GPU; --standalone picks a free port for the rendezvous
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(ids)}"] + train
    else:
        cmd = [sys.executable, "-u"] + train
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if ids:  # ids as nvidia-smi prints them
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(ids))
    print(summary)
    print(f"config:  {out}")
    print(f"results: {run_dir}  (best.pth, metrics.csv, metrics.png, log.txt)")
    print("command: " + (f"CUDA_VISIBLE_DEVICES={','.join(ids)} " if ids else "") + " ".join(cmd))
    if args.train:
        if ddp and os.name == "nt":
            sys.exit("DDP: torchrun from the Windows builds of torch cannot start its store (they lack libuv); "
                     "train on several GPUs under Linux, or set one device")
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "train.log", "a", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            sys.exit(proc.wait())


if __name__ == "__main__":
    main()
