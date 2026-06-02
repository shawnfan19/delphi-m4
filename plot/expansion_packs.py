"""Per-pack token-count histograms + pack-vs-disease co-occurrence heatmaps.

Dataset-agnostic — auto-detects UKB or AoU via DELPHI_DATASET /
delphi.data.auto.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from delphi.data.auto import detect_dataset, multimodal_reader_cls
from delphi.eval.cluster import CooccurrenceTracker
from delphi.experiment import CliConfig

mpl.rcParams["figure.dpi"] = 300

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "expansion_packs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    packs: list[str] = field(default_factory=list)


args = TaskConfig.from_cli()
args.print()

mm_cls = multimodal_reader_cls()
dataset_name = os.environ.get("DELPHI_DATASET") or detect_dataset()
n_total_pids = len(mm_cls.reader_cls.participants("all"))

pack_names = args.packs or mm_cls.expansion_pack_cls.catalog()
for pack_name in pack_names:
    reader = mm_cls(expansion_packs=[pack_name], memmap=True)
    pack = reader.expansion_packs[pack_name]

    # Histogram: tokens per participant
    pack_tokens_per_sub = np.array([pack.seq_len[int(p)] for p in pack.pids])
    plt.figure()
    plt.hist(pack_tokens_per_sub, bins="auto")
    plt.xlabel(f"{pack_name} tokens per participant")
    plt.ylabel("count")
    plt.title(f"{dataset_name}/{pack_name} — {len(pack.pids)} participants")
    out_path = OUT_DIR / f"{pack_name}_hist.png"
    with out_path.open("wb") as f:
        plt.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

    # Co-occurrence heatmap: pack tokens × disease tokens
    whitelist_keys = ["padding", "no_event"] + mm_cls.sex_keys + mm_cls.lifestyle_keys
    whitelist = np.array(
        [reader.tokenizer[k] for k in whitelist_keys if k in reader.tokenizer]
    )

    tracker = CooccurrenceTracker(vocab_size=reader.vocab_size)
    for pid in tqdm(pack.pids, desc=pack_name):
        x, t, *_ = reader[int(pid)]
        masked = np.where(np.isin(x, whitelist), 0, x)
        tracker.step(masked[None, :], t[None, :])
    cooccur = tracker.finalize()

    pack_ids = np.array(sorted(reader.expansion_tokens))
    base_ids = np.array(list(reader.base_tokenizer.values()))
    disease_ids = np.setdiff1d(base_ids, whitelist)
    heatmap = cooccur[np.ix_(pack_ids, disease_ids)] / n_total_pids

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
    plt.ylabel(f"{pack_name} token index")
    plt.title(f"{dataset_name}/{pack_name} × disease  ({len(pack.pids)} pids in pack)")
    out_path = OUT_DIR / f"{pack_name}_cooccur.png"
    with out_path.open("wb") as f:
        plt.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()
