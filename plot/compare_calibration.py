import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import yaml

from delphi.env import DELPHI_CKPT_READ
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

    def __post_init__(self):
        self.panel = parse_panel(self.panel)


args = TaskConfig.from_cli()
args.print()

with open(Path(DELPHI_CKPT_READ) / args.logbook, "r") as f:
    logbook = json.load(f)

time_horizon = list(logbook.keys())
colors = plt.cm.tab20(np.linspace(0, 1, len(time_horizon)))
for _, token in enumerate(args.panel):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    axs = axs.ravel()
    for i, sex in enumerate(["female", "male"]):
        axs[i].plot([0, 1], [0, 1], color="k")
        axs[i].set_title(sex)
        for j, horizon in enumerate(time_horizon):
            axs[i].scatter(
                logbook[horizon][token][sex]["pred"],
                logbook[horizon][token][sex]["obs"],
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
    # fig.legend(handles=legend_lines)
    fig.suptitle(token, fontsize=8)
    plt.show()


# legend_lines = list()
# for j, horizon in enumerate(time_horizon):
#     line = mlines.Line2D([], [], color=colors[j], label=f"{horizon / 365.25} years")
#     legend_lines.append(line)
