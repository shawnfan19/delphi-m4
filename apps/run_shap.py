# +
"""Token-only SHAP attribution for a (token-trained) DelphiM4 checkpoint.

Per validation patient, explain the model's next-event prediction at the last
history position with respect to each input TOKEN. Output is a per-patient
gzip pickle keyed by participant id: ``{pid: {"x", "t", "shap"}}`` plus
top-level ``tokenizer`` / ``model``.

Uses the current explainability primitives: ``ShapModel`` does the masking
(drop event -> reorder by time; mask last event -> no-event token to preserve
elapsed time; swap sex as a counterfactual) and ``ShapMasker`` is the
stateless pass-through the shap library drives. The custom masker has no
clustering, so shap dispatches to PermutationExplainer (plain Shapley).

Biomarker-trained models are NOT for this script — interpret those with the
saliency / integrated-gradients pipeline instead.
"""
import gzip
import pickle
import pprint
from dataclasses import dataclass

import numpy as np
import shap
from cloudpathlib import AnyPath
from tqdm import trange

from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.experiment import EvalConfig, load_ckpt, setup_eval_dataset
from delphi.explain.shap import ShapMasker, ShapModel


@dataclass(kw_only=True)
class TaskConfig(EvalConfig):
    fname_prefix = "shap"
    subsample: None | int = None
    half: bool = False
    # permutations per patient. shap's default (max_evals=500) silently caps long
    # sequences at ~3 permutations and ERRORS past ~250 tokens; pinning the count
    # via n_permutations * (2n+1) always clears shap's 2n+1 minimum. 10 == shap "auto".
    n_permutations: int = 10


args = TaskConfig.from_cli()
print("args:")
pprint.pp(args)
# -

# +
ckpt = AnyPath(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
if args.half:
    model.half()

# Token-only model: biomarker/expansion sets are empty in the ckpt, so the
# dataset yields empty bio_* and ShapModel attributes over tokens alone.
reader, ds, val_pids = setup_eval_dataset(
    ckpt_dict,
    fold=args.fold,
    override_biomarkers=args.biomarkers,
    override_expansion_packs=args.expansion_packs,
)
# -

# +
masker = ShapMasker()
total = len(ds) if args.subsample is None else min(args.subsample, len(ds))

# ponytail: whole dict held in RAM before the gzip dump (~vocab*float16 per token,
# summed over all patients -> several GB on full val). Stream per-pid pickles to one
# handle if that OOMs.
shap_pickle: dict = {}
for i in trange(total, leave=False):
    x0, t0, _, _, _, _, _ = ds[i]  # x0,t0 = history fed to the model
    pid = int(val_pids[i])
    n = len(x0)
    if n == 0:
        continue

    shap_model = ShapModel(model=model, data=(x0, t0))
    explainer = shap.Explainer(
        shap_model, masker, feature_names=np.arange(len(shap_model.dummy()))
    )
    shap_values = explainer(
        [shap_model.dummy()],
        max_evals=args.n_permutations * (2 * n + 1),  # type: ignore[arg-type]
        batch_size=args.batch_size,  # type: ignore[arg-type]
    )
    vals = shap_values.values[0].astype(np.float16)  # (n_tokens, vocab)

    # drop no-event tokens (id 1): masking leaves them unchanged -> ~0 SHAP
    keep = x0 != 1
    if not keep.any():
        continue
    shap_pickle[pid] = {
        "x": x0[keep],  # (n,) input token id
        "t": t0[keep],  # (n,) age in days at each token (anchor = t[-1])
        "shap": vals[keep],  # (n, vocab) float16; column j = token id j
    }
# -

# +
shap_pickle["tokenizer"] = ds.tokenizer  # name -> id
shap_pickle["model"] = str(ckpt)

out_dir = (AnyPath(DELPHI_CKPT_WRITE) / args.ckpt).parent
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{args.fname}.pickle.gz"
with gzip.open(out_path, "wb") as f:
    pickle.dump(shap_pickle, f)
print(f"Saved SHAP to {out_path}")
# -
