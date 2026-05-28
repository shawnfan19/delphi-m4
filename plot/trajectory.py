"""Print a randomly selected participant's trajectory as ASCII.

Events listed chronologically on the left; raw biomarker measurements (when
requested) on the right at the row matching their measurement age.
"""

from dataclasses import dataclass, field

import numpy as np

from delphi.data.auto import biomarker_cls, multimodal_reader_cls, reader_cls
from delphi.data.reader import TokenReader
from delphi.experiment import CliConfig


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    seed: int | None = None
    pid: int | None = None
    n: int = 1
    biomarkers: list[str] = field(default_factory=list)
    events_width: int = 26


args = TaskConfig.from_cli()
args.print()


def pick_pids(args: TaskConfig, reader_cls_, mm_cls) -> list[int]:
    if args.pid is not None:
        return [args.pid]

    pids = reader_cls_.participants("all")
    if args.biomarkers:
        pids = mm_cls.filter_participants_with_biomarkers(
            pids, biomarkers=args.biomarkers, any=True
        )
    rng = np.random.default_rng(args.seed)
    sampled = rng.choice(pids, size=args.n, replace=False)
    return [int(p) for p in sampled]


def build_rows(
    pid: int,
    reader: TokenReader,
    bios: dict[str, object],
    label_by_id: dict[int, str],
) -> tuple[str | None, list[tuple]]:
    tokens, times = reader[pid]

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
        label = label_by_id.get(tok_int) or reader.detokenizer[tok_int]
        rows.append((float(t), "event", label))

    for name, bio in bios.items():
        data, t_meas = bio[pid]
        if data is None:
            continue
        mod_name = name.upper()
        for vec, t in zip(data, t_meas):
            feats = [(f, float(v)) for f, v in zip(bio.features, vec)]
            rows.append((float(t), "visit", mod_name, feats))

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
            body.append(f"  {age:>3.0f}y  {label}")
        elif kind == "visit":
            _, _, mod, feats = row
            left = f"  {age:>3.0f}y  ─ {mod} ─".ljust(args.events_width)
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


reader_cls_ = reader_cls()
bio_cls = biomarker_cls()
mm_cls = multimodal_reader_cls()
reader = reader_cls_()
bios = {name: bio_cls(name) for name in args.biomarkers}

labels_fn = getattr(reader_cls_, "labels", None)
if labels_fn is not None:
    labels_df = labels_fn()
    label_by_id = {
        int(idx): str(name) for idx, name in zip(labels_df["index"], labels_df["name"])
    }
else:
    label_by_id = {}

pids = pick_pids(args, reader_cls_, mm_cls)

for pid in pids:
    sex, rows = build_rows(pid, reader, bios, label_by_id)
    render(pid, sex, rows, args)
    print()
