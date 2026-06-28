"""Per-pack token-count histograms + pack-vs-disease co-occurrence heatmaps.

Dataset-agnostic — auto-detects UKB or AoU via DELPHI_DATASET /
delphi.data.auto.
"""

import json
import os
from dataclasses import dataclass, field

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from cloudpathlib import AnyPath
from tqdm import tqdm

from delphi.data.auto import detect_dataset, multimodal_reader_cls
from delphi.env import DELPHI_RESULTS_DIR
from delphi.eval.cluster import CooccurrenceTracker, MissingDxTracker
from delphi.experiment import CliConfig
from delphi.plot import label_diseases

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    packs: list[str] = field(default_factory=list)
    write: str = "expansion_packs"
    # Dir of per-pack treatment-indication YAMLs (token -> diseases it treats).
    # Empty -> repo's data/ukb/dictionary/indication. A pack with a matching
    # <pack>.yaml gets the extra lead-time (Case A) scatter.
    indication_dir: str = ""
    # Min co-occurring participants for a (pack, disease) pair to enter the scatter.
    min_cooccur: int = 50


args = TaskConfig.from_cli()
args.print()

indication_dir = (
    AnyPath(args.indication_dir)
    if args.indication_dir
    else AnyPath(__file__).parents[2] / "data" / "ukb" / "dictionary" / "indication"
)

dataset_name = os.environ.get("DELPHI_DATASET") or detect_dataset()
OUT_DIR = AnyPath(DELPHI_RESULTS_DIR) / dataset_name / args.write
OUT_DIR.mkdir(parents=True, exist_ok=True)

mm_cls = multimodal_reader_cls()
base_pids = mm_cls.participants("all")

pack_names = args.packs or mm_cls.expansion_pack_cls.catalog()
for pack_name in pack_names:
    reader = mm_cls(expansion_packs=[pack_name])
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

    # Treatment-indication map for this pack (if any): pack-token id -> set of
    # disease-token ids it manages, fed to the Case B (missing-dx) tracker.
    ind_path = indication_dir / f"{pack_name}.yaml"
    indications = yaml.safe_load(ind_path.open()) if ind_path.exists() else None
    pack_to_dx = {
        reader.tokenizer[pt]: {
            reader.tokenizer[d] for d in entry["diseases"] if d in reader.tokenizer
        }
        for pt, entry in (indications or {}).items()
        if pt in reader.tokenizer
    }

    # One read pass feeds all three trackers: symmetric + directed co-occurrence
    # (Case A lead-time) and the missing-dx counter (Case B gap-filling).
    tracker = CooccurrenceTracker(vocab_size=reader.vocab_size)
    tracker_before = CooccurrenceTracker(vocab_size=reader.vocab_size, before=True)
    missing_dx = MissingDxTracker(pack_to_dx)
    for pid in tqdm(pack_pids, desc=pack_name):
        x, t, *_ = reader[int(pid)]
        masked = np.where(np.isin(x, whitelist), 0, x)
        tracker.step(masked, t)
        tracker_before.step(masked, t)
        missing_dx.step(x)

    pack_ids = np.array(sorted(reader.expansion_tokens))
    base_ids = np.array(list(reader.base_tokenizer.values()))
    disease_ids = np.setdiff1d(base_ids, whitelist)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [("symmetric", tracker), ("pack → disease", tracker_before)]
    for ax, (label, trk) in zip(axes, panels):
        heatmap = trk.finalize()[np.ix_(pack_ids, disease_ids)] / len(pack_pids)
        vmax = np.percentile(heatmap, 99.5)
        im = ax.imshow(
            np.log1p(heatmap),
            aspect="auto",
            cmap="inferno",
            vmin=0,
            vmax=np.log1p(vmax),
        )
        ax.set_xlabel("disease token index")
        ax.set_ylabel(f"{pack_name} token index")
        ax.set_title(label)
        fig.colorbar(
            im, ax=ax, label="log1p(co-occurring participants / pack participant)"
        )
    fig.suptitle(
        f"{dataset_name}/{pack_name} × disease  ({len(pack_pids)} pack participants)"
    )
    out_path = OUT_DIR / f"{pack_name}_cooccur.png"
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

    # --- Case A (lead-time): does a pack token precede the disease it treats? -
    # For each (pack token p, disease d) the directed/symmetric participant
    # counts give before/M_sym = fraction of co-occurring patients in whom p's
    # first occurrence precedes d's. Treatment pairs (from the indication file)
    # are highlighted against all co-occurring pairs (gray); if the drug/surgery
    # leads its diagnosis, the colored points sit above the gray cloud.
    if not indications:
        continue
    sym = tracker.finalize()[np.ix_(pack_ids, disease_ids)].astype(float)
    bef = tracker_before.finalize()[np.ix_(pack_ids, disease_ids)].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(sym > 0, bef / sym, np.nan)

    row_of = {int(t): i for i, t in enumerate(pack_ids)}
    col_of = {int(t): j for j, t in enumerate(disease_ids)}

    mapped = np.zeros(sym.shape, dtype=bool)
    rows = []
    for ptok, entry in indications.items():
        pid = reader.tokenizer.get(ptok)
        if pid not in row_of:
            continue
        for dtok in entry["diseases"]:
            did = reader.tokenizer.get(dtok)
            if did not in col_of:
                continue
            i, j = row_of[pid], col_of[did]
            mapped[i, j] = True
            rows.append((ptok, dtok, sym[i, j], bef[i, j], frac[i, j]))

    m = args.min_cooccur
    gray = (sym >= m) & ~mapped
    mdf = pd.DataFrame(
        rows, columns=["pack", "disease", "M_sym", "before", "before_frac"]
    )
    n_map_all = len(mdf)
    mdf = mdf[mdf["M_sym"] >= m].copy()
    mdf = label_diseases(mdf, key_col="disease")  # adds disease_name, color
    print(
        f"{pack_name}: {len(mdf)}/{n_map_all} treatment pairs with >= {m} "
        f"co-occurring participants; {int(gray.sum())} other pairs"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        sym[gray],
        frac[gray],
        s=6,
        c="#cccccc",
        alpha=0.4,
        linewidths=0,
        rasterized=True,
        label="other co-occurring pairs",
    )
    ax.scatter(
        mdf["M_sym"],
        mdf["before_frac"],
        s=28,
        c=mdf["color"].to_list(),
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
        label="treatment pairs",
    )
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"co-occurring participants  $M_{sym}[p,d]$  (log)")
    ax.set_ylabel(r"P(pack token before disease)  =  before / $M_{sym}$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"{dataset_name}/{pack_name}: lead-time of pack tokens vs the disease "
        f"they treat\n(>= {m} co-occurring participants, {len(mdf)} treatment pairs)"
    )
    for _, r in mdf.nlargest(15, "M_sym").iterrows():
        ax.annotate(
            f"{r['pack'].split('_', 1)[-1][:16]}→{r['disease'].split('_', 1)[0].upper()}",
            (r["M_sym"], r["before_frac"]),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.legend(loc="lower right", fontsize=8)
    out_path = OUT_DIR / f"{pack_name}_lead_time.png"
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

    out_json = OUT_DIR / f"{pack_name}_lead_time.json"
    with out_json.open("w") as f:
        json.dump(
            mdf[["pack", "disease", "M_sym", "before", "before_frac"]].to_dict(
                "records"
            ),
            f,
        )
    print(f"Saved {out_json}")

    # --- Case B (gap-filling): pack token present but none of its managed dx --
    # gap_frac = P(a holder of the token has NONE of its managed diseases coded).
    # High gap (for a tightly-mapped token) => the diagnosis was never recorded,
    # so the token is the only signal. It is an UPPER bound on true under-coding
    # (an incomplete dx set inflates it); colored by mapping breadth as a cue.
    counts = missing_dx.finalize()
    rows = [
        (reader.detokenizer[p], n, g / n, len(pack_to_dx[p]))
        for p, (n, g) in counts.items()
        if n >= m
    ]
    gdf = pd.DataFrame(rows, columns=["pack", "n_holders", "gap_frac", "n_diseases"])
    print(
        f"{pack_name}: {len(gdf)} tokens with >= {m} holders; "
        f"median gap_frac={gdf['gap_frac'].median():.2f}"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        gdf["n_holders"],
        gdf["gap_frac"],
        c=gdf["n_diseases"],
        cmap="viridis_r",
        s=28,
        edgecolors="black",
        linewidths=0.3,
    )
    fig.colorbar(sc, ax=ax, label="# diseases mapped (breadth; 1 = tightest)")
    ax.set_xscale("log")
    ax.set_xlabel("participants with the pack token (log)")
    ax.set_ylabel("Case B gap = P(token present & none of its managed diseases)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"{dataset_name}/{pack_name}: Case B — token present but no managed "
        f"disease coded\n({len(gdf)} tokens, >= {m} holders; gap is an upper "
        f"bound on under-coding)"
    )
    for _, r in gdf.nlargest(15, "n_holders").iterrows():
        ax.annotate(
            r["pack"].split("_", 1)[-1][:16],
            (r["n_holders"], r["gap_frac"]),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    out_path = OUT_DIR / f"{pack_name}_gap.png"
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

    out_json = OUT_DIR / f"{pack_name}_gap.json"
    with out_json.open("w") as f:
        json.dump(gdf.to_dict("records"), f)
    print(f"Saved {out_json}")
