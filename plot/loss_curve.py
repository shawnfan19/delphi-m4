"""Plot loss curves from one or more runs' stdout logs, overlaid for comparison.

Pass run paths RELATIVE to ``$DELPHI_CKPT_DIR``; with the offline-wandb layout a
run looks like ``wandb/offline-run-<ts>-<id>`` and its training stdout is at
``<run>/files/output.log``. Each run is parsed and overlaid (train loss faint,
val loss prominent), so you can compare runs on the AoU Workbench, where offline
wandb runs can't reach wandb.ai to visualize.

Self-contained on purpose: no ``delphi`` import, just stdlib + matplotlib +
cloudpathlib. It lives next to the print format it parses
(``delphi/experiment.py``) so it stays in sync, but it's one file you can copy
out to run standalone.

    python plot/loss_curve.py --runs wandb/offline-run-A wandb/offline-run-B
    python plot/loss_curve.py --runs <relpath> --out cmp.png --title "tiebreak sweep"
    python plot/loss_curve.py --demo      # parse self-check
"""

import argparse
import os
import re

# Print formats emitted by delphi/experiment.py BaseTrainer.train:
#   every log_interval:  "iter {i}: loss {x:.4f}"
#   every eval_interval: "iter {i}: train loss {x:.4f}, val loss {y:.4f}"
TRAIN_RE = re.compile(r"iter (\d+): loss ([\d.]+)")
EVAL_RE = re.compile(r"iter (\d+): train loss ([\d.]+), val loss ([\d.]+)")


def parse_log(text: str):
    """Return (train, val), each a list of (iter, loss), from a run's stdout."""
    train, val = [], []
    for line in text.splitlines():
        m = EVAL_RE.search(line)
        if m:
            val.append((int(m.group(1)), float(m.group(3))))
            continue
        m = TRAIN_RE.search(line)
        if m:
            train.append((int(m.group(1)), float(m.group(2))))
    return train, val


def plot(runs, out, title):
    """runs: list of (label, train, val). Train faint, val prominent, one color/run."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, train, val in runs:
        color, line = None, None
        if train:
            (line,) = ax.plot(*zip(*train), lw=1, alpha=0.4)
            color = line.get_color()
        if val:
            ax.plot(*zip(*val), "o-", lw=1.5, ms=3, color=color, label=label)
        elif line is not None:
            line.set_label(label)
    ax.set_xlabel("iter")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}  ({len(runs)} run(s))")


def _demo():
    sample = "\n".join(
        [
            "iter 0: loss 3.0000",
            "iter 0: train loss 3.0000, val loss 3.1000",
            "iter 250: loss 2.5000",
            "iter 500: loss 2.0000",
            "iter 500: train loss 2.0000, val loss 2.2000",
        ]
    )
    train, val = parse_log(sample)
    assert train == [(0, 3.0), (250, 2.5), (500, 2.0)], train
    assert val == [(0, 3.1), (500, 2.2)], val
    print("demo OK")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        nargs="+",
        default=[],
        help="run paths relative to --ckpt-dir, e.g. wandb/offline-run-<ts>-<id>",
    )
    p.add_argument(
        "--ckpt-dir",
        default=os.environ.get("DELPHI_CKPT_DIR", ""),
        help="root the --runs are relative to (default: $DELPHI_CKPT_DIR)",
    )
    p.add_argument("--out", default="loss.png", help="output PNG path")
    p.add_argument("--title", default="loss", help="plot title")
    p.add_argument("--demo", action="store_true", help="parse self-check and exit")
    args = p.parse_args()

    if args.demo:
        _demo()
        return
    if not args.runs:
        print("pass --runs <relpath> [<relpath> ...]  (or --demo)")
        return
    if not args.ckpt_dir:
        print("set $DELPHI_CKPT_DIR or pass --ckpt-dir")
        return

    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG, no display needed
    from cloudpathlib import AnyPath

    root = AnyPath(args.ckpt_dir)
    runs = []
    for rp in args.runs:
        log = root / rp / "files" / "output.log"
        if not log.exists():
            print(f"skip {rp}: no output.log at {log}")
            continue
        train, val = parse_log(log.read_text())
        if not train and not val:
            print(f"skip {rp}: no loss lines in output.log")
            continue
        runs.append((os.path.basename(rp.rstrip("/")), train, val))
    if runs:
        plot(runs, args.out, args.title)
    else:
        print("nothing to plot")


if __name__ == "__main__":
    main()
