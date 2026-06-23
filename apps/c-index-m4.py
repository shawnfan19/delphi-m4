# +
import json
import math
import pprint
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from cloudpathlib import AnyPath
from tqdm import tqdm

from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval import (
    ConcordanceCollator,
    DiseaseRatesCollator,
    EventTimeCollator,
)
from delphi.experiment import (
    EvalConfig,
    eval_iter,
    load_ckpt,
    move_batch_to_device,
    setup_eval_dataset,
)
from delphi.model.tpp import tpp_dispatch


@dataclass(kw_only=True)
class TaskConfig(EvalConfig):
    fname_prefix = "cindex"
    chunk_size: int = 8192
    max_gap: float = 5
    same_sex: bool = True


args = TaskConfig.from_cli()
print("args:")
pprint.pp(args)


# +
model, ckpt_dict = load_ckpt(AnyPath(DELPHI_CKPT_READ) / args.ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
reader, ds, val_pids = setup_eval_dataset(
    ckpt_dict,
    fold=args.fold,
    override_biomarkers=args.biomarkers,
    override_expansion_packs=args.expansion_packs,
)
# -

# +
offset_days = args.offset * 365.25
model_targets = model.targets.to(device)
# model.targets is the loss-scored set, not the disease set: exclude augmentation
# tokens (no_event, and the dx cluster anchor on tiebreak checkpoints) so they are
# never scored/ranked as diseases.
model_targets = model_targets[
    ~torch.isin(model_targets, model.augmentation_tokens.to(device))
]

dis_collator = DiseaseRatesCollator(targets=model_targets)
onset_collator = EventTimeCollator(vocab_size=int(model.config.vocab_size))

it = tqdm(
    eval_iter(total_size=len(ds), batch_size=args.batch_size),
    total=math.ceil(len(ds) / args.batch_size),
    desc="Phase 1",
    leave=False,
)
with torch.no_grad():
    for batch_idx in it:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        x0, t0, _, _, _, x1, t1 = batch_input

        out_dict, _, _ = model(*batch_input[:5])
        tpp = tpp_dispatch(model, out_dict)

        intensity, nearest_t0 = tpp.intensity(t1 - offset_days)

        dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=intensity)
        onset_collator.step(tokens=x1.cpu(), timestep=t1.cpu())

dis_rates, _ = dis_collator.finalize()  # (N, V)
is_female = torch.from_numpy(reader.is_female(val_pids))  # (N,)
onset_times, _ = onset_collator.finalize()
onset_times = torch.from_numpy(onset_times)  # (N, V)

# Move tensors to device for Phase 2
dis_rates = dis_rates.to(device)
onset_times = onset_times.to(device)
is_female = is_female.to(device)

# +
concordance_collator = ConcordanceCollator(
    dis_rates=dis_rates,
    case_times=onset_times - offset_days,
    is_female=is_female,
    max_gap_days=args.max_gap * 365.25,
    chunk_size=args.chunk_size,
    same_sex_only=args.same_sex,
)

# dis_rates and onset_times are fully consumed above (the collator extracts its
# case events and keeps its own case_times_mat copy); free both (N, V) matrices
# and return Phase 1's reserved pool before the Phase 2 forward passes.
del dis_rates, onset_times
if device == "cuda":
    torch.cuda.empty_cache()

it2 = tqdm(
    eval_iter(total_size=len(ds), batch_size=args.batch_size),
    total=math.ceil(len(ds) / args.batch_size),
    desc="Phase 2",
    leave=False,
)
with torch.no_grad():
    for batch_idx in it2:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        out_dict, _, _ = model(*batch_input[:5])
        tpp = tpp_dispatch(model, out_dict)
        concordance_collator.step(tpp=tpp)

case_sex, case_tokens, total_pairs, concordant = concordance_collator.finalize()
case_times = concordance_collator.case_times.cpu().numpy()
case_participants = concordance_collator.case_participants.cpu().numpy()
# -

out_dir = (AnyPath(DELPHI_CKPT_WRITE) / args.ckpt).parent
out_dir.mkdir(parents=True, exist_ok=True)

pids_np = np.array(val_pids)
cindex_df = pd.DataFrame(
    {
        "icd": pd.Categorical(
            [reader.detokenizer.get(int(d), str(d)) for d in case_tokens]
        ),
        "sex": pd.Categorical(np.where(case_sex, "female", "male")),
        "participant_id": pids_np[case_participants].astype(np.int64),
        "case_time": case_times.astype(np.float32),
        "concordant": concordant.astype(np.float32),
        "total_pairs": total_pairs.astype(np.int32),
    }
)
# Embed the run config in the Parquet footer so it travels inside the file.
table = pa.Table.from_pandas(cindex_df, preserve_index=False)
table = table.replace_schema_metadata(
    {
        **(table.schema.metadata or {}),
        b"config": json.dumps(asdict(args), default=str).encode(),
    }
)
cindex_path = out_dir / f"{args.fname}.parquet"
with cindex_path.open("wb") as f:
    pq.write_table(table, f, compression="snappy")
print(f"Saved c-index to {cindex_path}")
