"""Dataset overview: tokens-per-participant and disease-cluster histograms.

Dataset-agnostic — loads the UKB or AoU reader via delphi.data.auto.
"""

import os
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cloudpathlib import AnyPath
from tqdm import tqdm

from delphi.data.auto import detect_dataset, multimodal_reader_cls
from delphi.env import DELPHI_RESULTS_DIR
from delphi.eval.cluster import ClusterStatsTracker, TiedEventTracker
from delphi.experiment import CliConfig

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    write: str = ""  # optional subdir under DELPHI_RESULTS_DIR


args = TaskConfig.from_cli()
args.print()

mm_cls = multimodal_reader_cls()
reader = mm_cls()
dataset_name = os.environ.get("DELPHI_DATASET") or detect_dataset()
OUT_DIR = AnyPath(DELPHI_RESULTS_DIR) / args.write / dataset_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

pids = mm_cls.participants("all")
print(f"{dataset_name}: {len(pids)} participants")

tokens_per_sub = np.array([reader.token_reader.seq_len[int(p)] for p in pids])

whitelist_keys = ["padding", "no_event"] + reader.sex_keys + reader.lifestyle_keys
whitelist = np.array(
    [reader.tokenizer[k] for k in whitelist_keys if k in reader.tokenizer]
)

tracker = ClusterStatsTracker()
cooccur_tracker = TiedEventTracker(vocab_size=reader.vocab_size)
for pid in tqdm(pids):
    tokens, times = reader.token_reader[int(pid)]
    masked = np.where(np.isin(tokens, whitelist), 0, tokens)
    tracker.step(masked[None, :], times[None, :])
    cooccur_tracker.step(masked[None, :], times[None, :])

n_clusters_per_sub, cluster_sizes = tracker.finalize()
cluster_sizes = cluster_sizes[cluster_sizes > 1]
cooccur = cooccur_tracker.finalize()
disease_ids = np.setdiff1d(np.array(list(reader.tokenizer.values())), whitelist)
heatmap = cooccur[np.ix_(disease_ids, disease_ids)] / len(pids)


plt.figure()
plt.hist(tokens_per_sub, bins="auto")
plt.xlabel("tokens per participant")
plt.ylabel("count")
plt.title(f"{dataset_name} — {len(pids)} participants")
out_path = OUT_DIR / "tokens_per_sub.png"
with out_path.open("wb") as f:
    plt.savefig(f, format="png", bbox_inches="tight")
print(f"Saved {out_path}")
plt.close()

plt.figure()
plt.hist(n_clusters_per_sub, bins="auto")
plt.xlabel("disease clusters per participant")
plt.ylabel("count")
plt.title(f"{dataset_name} — {len(pids)} participants")
out_path = OUT_DIR / "n_clusters_per_sub.png"
with out_path.open("wb") as f:
    plt.savefig(f, format="png", bbox_inches="tight")
print(f"Saved {out_path}")
plt.close()

plt.figure()
plt.hist(cluster_sizes, bins="auto")
plt.xlabel("cluster size (tokens per day, clusters of size >1)")
plt.ylabel("count")
plt.title(f"{dataset_name} — {len(pids)} participants")
out_path = OUT_DIR / "cluster_sizes.png"
with out_path.open("wb") as f:
    plt.savefig(f, format="png", bbox_inches="tight")
print(f"Saved {out_path}")
plt.close()

vmax = np.percentile(heatmap, 99.5)
plt.figure()
plt.imshow(
    np.log1p(heatmap),
    aspect="auto",
    cmap="inferno",
    vmin=0,
    vmax=np.log1p(vmax),
)
plt.colorbar(label="log1p(events / participant)")
plt.xlabel("disease token index")
plt.ylabel("disease token index")
plt.title(f"{dataset_name} disease × disease — {len(pids)} participants")
out_path = OUT_DIR / "disease_cooccur.png"
with out_path.open("wb") as f:
    plt.savefig(f, format="png", bbox_inches="tight")
print(f"Saved {out_path}")
plt.close()
