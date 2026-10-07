"""
Per-epoch metrics of a training run: appends <output_dir>/metrics.csv and redraws <output_dir>/metrics.png.

Precision / recall / F1 are read from the COCO PR curves at IoU 0.5 (all areas, maxDets 100): for every class the
point of its curve with the best F1 is taken, then the values are averaged over classes that have ground truth.
"""

import csv
from pathlib import Path

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def pr_at_best_f1(coco_eval):
    prec = np.asarray(coco_eval.eval["precision"])[0, :, :, 0, -1]  # [recall thresholds, classes] at IoU 0.5
    rec = np.asarray(coco_eval.params.recThrs)
    ps, rs, fs, ap50 = [], [], [], []
    for k in range(prec.shape[1]):
        p = prec[:, k]
        if (p < 0).all():  # class without ground truth
            ap50.append(float("nan"))
            continue
        p = np.clip(p, 0, None)
        f1 = 2 * p * rec / np.maximum(p + rec, 1e-12)
        i = int(f1.argmax())
        ps.append(p[i])
        rs.append(rec[i])
        fs.append(f1[i])
        ap50.append(float(p.mean()))
    mean = (lambda v: float(np.mean(v)) if v else float("nan"))
    return mean(ps), mean(rs), mean(fs), ap50


def update(output_dir, epoch, train_stats, test_stats, coco_evaluator, head_lr=None):
    output_dir = Path(output_dir)
    coco_eval = coco_evaluator.coco_eval["bbox"]
    stats = test_stats["coco_eval_bbox"]
    p, r, f1, ap50 = pr_at_best_f1(coco_eval)
    gt = coco_eval.cocoGt
    names = [c["name"] for c in gt.loadCats(sorted(gt.getCatIds()))]
    lr = head_lr if head_lr is not None else train_stats.get("lr", float("nan"))
    row = {"epoch": epoch + 1, "lr": lr, "train_loss": train_stats.get("loss"),
           "map50": stats[1], "map50_95": stats[0], "precision": p, "recall": r, "f1": f1, "ar100": stats[8],
           **{f"ap50_{n}": v for n, v in zip(names, ap50)}}
    path = output_dir / "metrics.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    plot(output_dir)


def plot(output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_dir = Path(output_dir)
    rows = list(csv.DictReader((output_dir / "metrics.csv").open()))
    ep = [int(r["epoch"]) for r in rows]
    col = (lambda k: [float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
    cls = [k for k in rows[0] if k.startswith("ap50_")]

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": GRID})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=110, facecolor=SURFACE)
    panels = [
        ("mAP (val)", [("mAP@0.5", "map50"), ("mAP@0.5:0.95", "map50_95")], (0, 1)),
        ("precision / recall / F1 (val, IoU 0.5, best-F1 point)",
         [("precision", "precision"), ("recall", "recall"), ("F1", "f1")], (0, 1)),
        ("AP@0.5 by class (val)", [(k[5:], k) for k in cls], (0, 1)),
        ("train loss", [("loss", "train_loss")], None),
        ("learning rate (head)", [("lr", "lr")], None),
    ]
    for ax, (title, series, ylim) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for i, (label, key) in enumerate(series[: len(PALETTE)]):
            ax.plot(ep, col(key), "-o", color=PALETTE[i], linewidth=1.6, markersize=3.5, label=label)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if ylim:
            ax.set_ylim(*ylim)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=8.5)
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics.png", facecolor=SURFACE)
    plt.close(fig)
