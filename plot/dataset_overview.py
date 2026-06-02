"""Dataset overview: tokens-per-participant and disease-cluster histograms.

Dataset-agnostic — loads the UKB or AoU reader via delphi.data.auto.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from delphi.data.auto import detect_dataset, multimodal_reader_cls
from delphi.eval.cluster import ClusterStatsTracker

mpl.rcParams["figure.dpi"] = 300


mm_cls = multimodal_reader_cls()
reader = mm_cls.reader_cls()
dataset_name = os.environ.get("DELPHI_DATASET") or detect_dataset()

pids = mm_cls.reader_cls.participants("all")
print(f"{dataset_name}: {len(pids)} participants")

tokens_per_sub = np.array([reader.seq_len[int(p)] for p in pids])

whitelist_keys = ["padding", "no_event"] + reader.sex_keys + reader.lifestyle_keys
whitelist = np.array(
    [reader.tokenizer[k] for k in whitelist_keys if k in reader.tokenizer]
)

tracker = ClusterStatsTracker()
for pid in tqdm(pids):
    tokens, times = reader[int(pid)]
    masked = np.where(np.isin(tokens, whitelist), 0, tokens)
    tracker.step(masked[None, :], times[None, :])

n_clusters_per_sub, cluster_sizes = tracker.finalize()
cluster_sizes = cluster_sizes[cluster_sizes > 1]


plt.figure()
plt.hist(tokens_per_sub, bins="auto")
plt.xlabel("tokens per participant")
plt.ylabel("count")
plt.title(f"{dataset_name} — {len(pids)} participants")
plt.show()

plt.figure()
plt.hist(n_clusters_per_sub, bins="auto")
plt.xlabel("disease clusters per participant")
plt.ylabel("count")
plt.title(f"{dataset_name} — {len(pids)} participants")
plt.show()

plt.figure()
plt.hist(cluster_sizes, bins="auto")
plt.xlabel("cluster size (tokens per day, clusters of size >1)")
plt.ylabel("count")
plt.title(f"{dataset_name} — {len(pids)} participants")
plt.show()
