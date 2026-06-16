import copy
import json
import os
import pprint
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cloudpathlib import AnyPath

from delphi.env import DELPHI_CKPT_READ as DELPHI_CKPT_DIR
from delphi.experiment import CliConfig, flexi_list, load_json


def plot_concordance(ax, concord, t, bin_size, color="steelblue", label=None):
    ax.scatter(
        t,
        concord,
        c=color,
        s=40,
        alpha=0.1,
        edgecolors="black",
        linewidths=0.5,
        zorder=2,
    )

    concord_mu = np.bincount(np.floor(t / bin_size).astype(int), weights=concord)
    per_bin_count = np.bincount(np.floor(t / bin_size).astype(int))
    concord_mu /= per_bin_count
    t_mu = np.arange(len(concord_mu)) * bin_size
    is_valid = ~np.isnan(concord_mu)
    ax.plot(t_mu[is_valid], concord_mu[is_valid], label=label)


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    doi: Any
    json: str
    baseline_json: str
    bin_size: int = 5

    def __post_init__(self):
        self.doi = flexi_list(self.doi)


args = TaskConfig.from_cli()
args.print()

json_path = AnyPath(DELPHI_CKPT_DIR) / args.json
baseline_json_path = AnyPath(DELPHI_CKPT_DIR) / args.baseline_json

data_a, _ = load_json(json_path)
data_b, _ = load_json(baseline_json_path)

colors = plt.cm.tab20(np.linspace(0, 1, 2))
for disease in args.doi:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    for ax, sex in zip(axes, ["female", "male"]):
        concord = np.array(data_a[disease][sex]["c_index_by_t"])
        t = np.array(data_a[disease][sex]["t"]) / 365.25
        plot_concordance(
            ax, concord, t, bin_size=args.bin_size, label=json_path.parent.stem
        )

        concord = np.array(data_b[disease][sex]["c_index_by_t"])
        t = np.array(data_b[disease][sex]["t"]) / 365.25
        plot_concordance(
            ax, concord, t, bin_size=args.bin_size, label=baseline_json_path.parent.stem
        )

        ax.set_xlim(0, 100)
        ax.set_ylim(0.5, 1.0)
        ax.set_xlabel("time")
        ax.set_title(sex)
        ax.legend()
    fig.suptitle(disease)
    plt.show()
