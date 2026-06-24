import json
import math
from dataclasses import dataclass, field
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import yaml
from cloudpathlib import AnyPath

from delphi.env import DELPHI_CKPT_READ, DELPHI_RESULTS_DIR
from delphi.experiment import CliConfig


def parse_panel(panel):
    if isinstance(panel, str):
        if panel.endswith(".yaml"):
            with open(panel, "r") as f:
                return yaml.safe_load(f)
        else:
            return [panel]
    elif isinstance(panel, list):
        return panel
    else:
        raise ValueError


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    logbook: str
    panel: Any
    # Columns in the disease-grid overview figure (rows are derived).
    ncols: int = 5
    # If set, also save figures under DELPHI_RESULTS_DIR/<write>/.
    write: None | str = None

    def __post_init__(self):
        self.panel = parse_panel(self.panel)


args = TaskConfig.from_cli()
args.print()

with (AnyPath(DELPHI_CKPT_READ) / args.logbook).open("r") as f:
    logbook = json.load(f)

OUT_DIR = None
if args.write is not None:
    OUT_DIR = AnyPath(DELPHI_RESULTS_DIR) / args.write
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    if OUT_DIR is None:
        return
    out_path = OUT_DIR / name
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")


def either_calibration(entry):
    """Sex-pooled per-bin (pred, obs) from a logbook entry's female & male.

    The logbook only stores ``female``/``male``; "either" is synthesized as the
    per-bin ``counts``-weighted mean of the two. ``pred`` is an exact
    population-weighted mean; ``obs`` is the count-weighted mean of the per-sex
    KM incidences (an approximation of the gender-pooled incidence, matching the
    "either" convention used for AUC elsewhere). Empty bins carry zero weight,
    and bins with zero total count become NaN so they aren't drawn.
    """
    f, m = entry["female"], entry["male"]
    w_f = np.asarray(f["counts"], dtype=float)
    w_m = np.asarray(m["counts"], dtype=float)
    den = w_f + w_m
    out = []
    for field in ("pred", "obs"):
        v_f = np.asarray(f[field], dtype=float)
        v_m = np.asarray(m[field], dtype=float)
        # NaN values only occur in empty bins (weight 0); zero them so the
        # weighted sum ignores them rather than propagating NaN.
        num = np.nan_to_num(v_f) * w_f + np.nan_to_num(v_m) * w_m
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(den > 0, num / den, np.nan))
    return out  # [pred, obs]


time_horizon = list(logbook.keys())
colors = plt.cm.tab20(np.linspace(0, 1, len(time_horizon)))
for _, token in enumerate(args.panel):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    axs = axs.ravel()
    for i, sex in enumerate(["either", "female", "male"]):
        axs[i].plot([0, 1], [0, 1], color="k")
        axs[i].set_title(sex)
        for j, horizon in enumerate(time_horizon):
            entry = logbook[horizon][token]
            if sex == "either":
                pred, obs = either_calibration(entry)
            else:
                pred, obs = entry[sex]["pred"], entry[sex]["obs"]
            axs[i].scatter(
                pred,
                obs,
                color=colors[j],
                marker="o",
                alpha=0.7,
                label=horizon,
            )
            # axs[i].scatter(pred_mu, obs_mu, color=colors[j], marker="X")
        axs[i].legend()
    axs[-1].set_yscale("log")
    axs[-1].set_xscale("log")
    axs[-1].set_ylim(1e-5, 1)
    axs[-1].set_xlim(1e-5, 1)
    fig.supxlabel("predicted rates")
    fig.supylabel("observed rates", x=0.08)
    fig.suptitle(token, fontsize=8)
    save_fig(fig, f"calibration_{token.replace('/', '_')}.png")
    plt.show()


# Overview figure: one panel per --panel disease, showing only the sex-pooled
# ("either") calibration, each titled with the disease (token) name.
ncols = min(args.ncols, len(args.panel))
nrows = math.ceil(len(args.panel) / ncols)
fig, axs = plt.subplots(
    nrows, ncols, figsize=(3 * ncols, 3 * nrows), sharex=True, sharey=True
)
axs = np.atleast_1d(axs).ravel()
for k, token in enumerate(args.panel):
    ax = axs[k]
    ax.plot([0, 1], [0, 1], color="k")
    ax.set_title(token, fontsize=9)
    for j, horizon in enumerate(time_horizon):
        pred, obs = either_calibration(logbook[horizon][token])
        ax.scatter(pred, obs, color=colors[j], marker="o", alpha=0.7)
for ax in axs[len(args.panel) :]:
    ax.set_visible(False)
# Shared log-log scale propagates to every panel via sharex/sharey.
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlim(1e-5, 1)
axs[0].set_ylim(1e-5, 1)
fig.supxlabel("predicted rates")
fig.supylabel("observed rates")
fig.suptitle("either-sex calibration by disease")
# One shared horizon legend in the reserved right margin.
handles = [
    mlines.Line2D([], [], color=colors[j], marker="o", linestyle="", label=str(horizon))
    for j, horizon in enumerate(time_horizon)
]
fig.tight_layout(rect=(0, 0, 0.88, 0.96))
fig.legend(
    handles=handles, title="horizon", loc="center right", bbox_to_anchor=(1.0, 0.5)
)
save_fig(fig, "calibration_overview.png")
plt.show()
