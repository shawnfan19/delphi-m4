from collections import defaultdict

import numpy as np
import shap
import torch

from delphi.data.utils import collate_batch
from delphi.multimodal import Modality


class ShapMasker(shap.maskers.Masker):  # type: ignore

    def __init__(self):
        self.immutable_outputs = True
        self.fixed_background = False

    def shape(self, s: tuple[np.ndarray, np.ndarray]):
        return (1, len(s[0]))

    def mask_shapes(self, s: tuple[np.ndarray, np.ndarray]):
        return [(len(s[0]),)]

    def __call__(self, mask, s: tuple[np.ndarray, np.ndarray]):
        mask = self._standardize_mask(mask, s)
        x, t = s
        x, t = x.copy(), t.copy()
        assert x.size == t.size
        l = x.size
        pos = np.arange(l)

        # mask: True means that a feature participates
        masked = ~mask
        is_event = (x > 3).copy()
        is_female = (x == 2).copy()
        is_male = (x == 3).copy()
        is_last = pos == l - 1

        t = np.where(is_event & masked & ~is_last, -1e4, t)
        x = np.where(is_event & masked & ~is_last, 0, x)

        # if the last token were to be masked, replace it with the no-event token
        # to avoid biasing the elapsed time estimate
        x = np.where(is_event & masked & is_last, 1, x)

        x = np.where(is_male & masked, 2, x)
        x = np.where(is_female & masked, 3, x)

        sort_idx = np.argsort(t)
        x = np.take_along_axis(x, sort_idx, axis=0)
        t = np.take_along_axis(t, sort_idx, axis=0)

        return ((x,), (t,))


@torch.no_grad
def shap_forward(
    idx: list[np.ndarray],
    age: list[np.ndarray],
    model: torch.nn.Module,
    doi: list[int],
):
    x_lst, t_lst = list(), list()
    for x, t in zip(idx, age):
        x_lst.append(x)
        t_lst.append(t)
    x = collate_batch(x_lst)
    t = collate_batch(t_lst)
    device = next(model.parameters()).device
    x = torch.tensor(x).to(device).long()
    t = torch.tensor(t).to(device)

    outputs, _, _ = model.forward(x, t)
    # for compatibility with legacy model definition
    if isinstance(outputs, torch.Tensor):
        logits = outputs
    else:
        logits = outputs["logits"]
    doi_logits = logits[:, -1, doi].detach().cpu().numpy()

    return doi_logits


MultimodalOut = tuple[
    np.ndarray, np.ndarray, dict[Modality, list[np.ndarray]], np.ndarray, np.ndarray
]


class MultimodalShapMasker(shap.maskers.Masker):  # type: ignore
    """SHAP masker for measurement-level attribution with missingness background.

    Each SHAP feature is one biomarker measurement (modality × time-point).
    Outputs a binary mask (1=present, 0=absent) consumed by multimodal_shap_forward.
    Tokens are not SHAP features; they are always present in the model input.
    """

    def __init__(self):
        self.immutable_outputs = True
        self.fixed_background = False

    def shape(self, bio_t: np.ndarray):
        return (1, len(bio_t))

    def mask_shapes(self, bio_t: np.ndarray):
        return [(len(bio_t),)]

    def __call__(self, mask, bio_t: np.ndarray):
        mask = self._standardize_mask(mask, bio_t)
        return ((mask.astype(np.int8),),)


@torch.no_grad
def multimodal_shap_forward(
    masks: list[np.ndarray],
    *,
    out: MultimodalOut,
    model,
):
    """Forward function for SHAP explainer operating directly on MultimodalOut.

    Args:
        masks: list of binary arrays (one per coalition), each of length n_measurements.
               1 = measurement present, 0 = absent. Provided by shap.Explainer.
        out:   MultimodalOut (x, t, bio_x_dict, bio_t, bio_m) for the participant
               being explained. Baked in via functools.partial.
        model: multimodal model with .forward(idx, age, biomarker, mod_age, mod_idx).
    """
    x, t, bio_x_dict, bio_t, bio_m = out

    x_lst, t_lst = [], []
    batched_bio_x: dict = defaultdict(list)
    bio_t_lst, bio_m_lst = [], []

    for mask in masks:
        mask_bool = mask.astype(bool)
        new_bio_t = bio_t[mask_bool]
        new_bio_m = bio_m[mask_bool]

        # Apply mask to bio_x_dict.
        # bio_x_dict[M][k] is the k-th measurement of modality M in chronological
        # order, matching the k-th occurrence of M in bio_m. We track per-modality
        # counters to select the right rows.
        per_mod_counter: dict = defaultdict(int)
        new_bio_x: dict = defaultdict(list)
        for present, mval in zip(mask_bool, bio_m):
            mod = Modality(int(mval))
            k = per_mod_counter[mod]
            per_mod_counter[mod] += 1
            if present:
                new_bio_x[mod].append(bio_x_dict[mod][k])

        x_lst.append(x)
        t_lst.append(t)
        for mod, arrays in new_bio_x.items():
            batched_bio_x[mod].extend(arrays)
        bio_t_lst.append(new_bio_t)
        bio_m_lst.append(new_bio_m)

    device = next(model.parameters()).device
    idx = collate_batch(x_lst)
    idx = torch.tensor(idx, dtype=torch.long).to(device)
    age = collate_batch(t_lst, fill_value=-1e4)
    age = torch.tensor(age, dtype=torch.float32).to(device)

    bio_x_batch = {}
    for mod, arrays in batched_bio_x.items():
        bio_x_batch[mod] = torch.from_numpy(np.stack(arrays)).to(device)  # type: ignore

    mod_age = collate_batch(bio_t_lst, fill_value=-1e4)
    mod_age = torch.tensor(mod_age, dtype=torch.float32).to(device)
    mod_idx = collate_batch(bio_m_lst)
    mod_idx = torch.tensor(mod_idx, dtype=torch.long).to(device)

    outputs, _, _ = model.forward(idx, age, bio_x_batch, mod_age, mod_idx)
    logits = outputs["logits"][:, -1, :]
    logits[:, model.config.ignore_tokens] = -torch.inf

    return logits.detach().cpu().numpy()
