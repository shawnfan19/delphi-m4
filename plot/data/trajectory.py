"""Print a randomly selected participant's trajectory as ASCII.

Events listed chronologically on the left; raw biomarker measurements (when
requested) on the right at the row matching their measurement age.
"""

from dataclasses import dataclass, field

import numpy as np

from delphi.data.auto import multimodal_reader_cls
from delphi.data.reader import MultimodalReader
from delphi.experiment import CliConfig


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    seed: int | None = None
    pid: int | None = None
    n: int = 1
    biomarkers: list[str] = field(default_factory=list)
    expansion_packs: list[str] = field(default_factory=list)
    events_width: int = 26


args = TaskConfig.from_cli()
args.print()


def pick_pids(args: TaskConfig, mm_cls) -> list[int]:
    if args.pid is not None:
        return [args.pid]

    pids = mm_cls.reader_cls.participants("all")
    if args.biomarkers or args.expansion_packs:
        pids = mm_cls.filter_participants_with_modalities(
            pids,
            biomarkers=args.biomarkers or None,
            expansion_packs=args.expansion_packs or None,
        )
    rng = np.random.default_rng(args.seed)
    sampled = rng.choice(pids, size=args.n, replace=False)
    return [int(p) for p in sampled]


def build_rows(
    pid: int,
    reader: MultimodalReader,
    pack_tag: dict[int, str],
) -> tuple[str | None, list[tuple]]:
    tokens, times, bio_x_dict, bio_t, bio_m = reader[pid]

    female_id = reader.tokenizer["female"]
    male_id = reader.tokenizer["male"]

    sex = None
    if (tokens == female_id).any():
        sex = "Female"
    elif (tokens == male_id).any():
        sex = "Male"

    rows: list[tuple] = []
    for tok, t in zip(tokens, times):
        tok_int = int(tok)
        if tok_int in (female_id, male_id):
            continue
        label = reader.detokenizer[tok_int]
        pack = pack_tag.get(tok_int)
        if pack:
            label = f"[{pack}] {label}"
        rows.append((float(t), "event", label))

    # Biomarker visits straight from the reader's outputs. bio_x_dict[name] holds the
    # measurement vectors in modality order; bio_t/bio_m are globally time-sorted, so
    # bio_t[bio_m == idx] gives that modality's ascending times, matching the vectors.
    # Feature names come from the static schema, not the participant's data.
    for name, vecs in bio_x_dict.items():
        t_mod = bio_t[bio_m == reader.biomarker2idx[name]]
        feat_names = reader.biomarkers[name].features
        for vec, t in zip(vecs, t_mod):
            feats = [(f, float(v)) for f, v in zip(feat_names, vec)]
            rows.append((float(t), "visit", name.upper(), feats))

    rows.sort(key=lambda r: r[0])
    return sex, rows


def render(
    pid: int,
    sex: str | None,
    rows: list[tuple],
    args: TaskConfig,
) -> None:
    has_bio = bool(args.biomarkers)
    blank_left = " " * args.events_width

    body: list[str] = []
    for row in rows:
        age = row[0] / 365.25
        kind = row[1]
        if kind == "event":
            _, _, label = row
            body.append(f"  {age:>6.2f}y  {label}")
        elif kind == "visit":
            _, _, mod, feats = row
            left = f"  {age:>6.2f}y  ─ {mod} ─".ljust(args.events_width)
            first_feat, first_val = feats[0]
            body.append(f"{left} │ {first_feat:<6}  {first_val:>8.3g}")
            for feat, val in feats[1:]:
                body.append(f"{blank_left} │ {feat:<6}  {val:>8.3g}")
        else:
            raise ValueError(f"unknown row kind: {kind}")

    if not body:
        print(f"Participant {pid}: no events.")
        return

    first_yr = rows[0][0] / 365.25
    last_yr = rows[-1][0] / 365.25
    sex_part = sex or "Unknown sex"
    header = f"Participant {pid} · {sex_part} · " f"age {first_yr:.0f}-{last_yr:.0f}y"

    col_header = None
    if has_bio:
        col_header = f"{'  EVENTS'.ljust(args.events_width)} │ BIOMARKERS (raw)"

    width = max(len(header), *(len(line) for line in body))
    if col_header is not None:
        width = max(width, len(col_header))

    print(header)
    print("═" * width)
    if col_header is not None:
        print(col_header)
    for line in body:
        print(line)
    print("═" * width)


mm_cls = multimodal_reader_cls()
reader = mm_cls(
    expansion_packs=args.expansion_packs,
    biomarkers=args.biomarkers,
)

# Map each merged-vocab token id back to its expansion pack, so pack tokens can be
# tagged. Pack-local ids are shifted by expansion_offset[name] when merged (see
# MultimodalReader.__getitem__); base tokens are absent from this map (no tag).
pack_tag = {
    local_id + reader.expansion_offset[name]: name
    for name, pack in reader.expansion_packs.items()
    for local_id in pack.tokenizer.values()
}

pids = pick_pids(args, mm_cls)

for pid in pids:
    sex, rows = build_rows(pid, reader, pack_tag)
    render(pid, sex, rows, args)
    print()
