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


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    packs: list[str] = field(default_factory=list)


args = TaskConfig.from_cli()
args.print()

dataset_name = os.environ.get("DELPHI_DATASET") or detect_dataset()
OUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "expansion_packs" / dataset_name
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

mm_cls = multimodal_reader_cls()
base_pids = mm_cls.reader_cls.participants("all")

pack_names = args.packs or mm_cls.expansion_pack_cls.catalog()
for pack_name in pack_names:
    reader = mm_cls(expansion_packs=[pack_name], memmap=True)
    pack = reader.expansion_packs[pack_name]

    # Intersect pack pids with the base reader's pids — drop orphan pids
    # that appear in the pack but not in the base data.
    keep = np.isin(pack.pids, base_pids)
    pack_pids = pack.pids[keep]
    dropped = int((~keep).sum())
    if dropped:
        print(f"{pack_name}: dropped {dropped} orphan pids not in base reader")

    # Histogram: tokens per participant
    pack_tokens_per_sub = np.array([pack.seq_len[int(p)] for p in pack_pids])
    plt.figure()
    plt.hist(pack_tokens_per_sub, bins="auto")
    plt.xlabel(f"{pack_name} tokens per participant")
    plt.ylabel("count")
    plt.title(f"{dataset_name}/{pack_name} — {len(pack_pids)} participants")
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
    for pid in tqdm(pack_pids, desc=pack_name):
        x, t, *_ = reader[int(pid)]
        masked = np.where(np.isin(x, whitelist), 0, x)
        tracker.step(masked[None, :], t[None, :])
    cooccur = tracker.finalize()

    pack_ids = np.array(sorted(reader.expansion_tokens))
    base_ids = np.array(list(reader.base_tokenizer.values()))
    disease_ids = np.setdiff1d(base_ids, whitelist)
    heatmap = cooccur[np.ix_(pack_ids, disease_ids)] / len(pack_pids)

    vmax = np.percentile(heatmap, 99.5)
    plt.figure()
    plt.imshow(
        np.log1p(heatmap),
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=np.log1p(vmax),
    )
    plt.colorbar(label="log1p(events / pack participant)")
    plt.xlabel("disease token index")
    plt.ylabel(f"{pack_name} token index")
    plt.title(
        f"{dataset_name}/{pack_name} × disease  "
        f"({len(pack_pids)} pack participants)"
    )
    out_path = OUT_DIR / f"{pack_name}_cooccur.png"
    with out_path.open("wb") as f:
        plt.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()
