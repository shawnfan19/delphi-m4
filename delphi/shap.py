from collections import defaultdict

import numpy as np
import shap
import torch

from delphi.data.utils import collate_batch, sort_by_time
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

ShapArray = tuple[np.ndarray, np.ndarray, np.ndarray]

# example:
# MultimodalOut:
#    x: [2, 101, 1269]
#    t: [0, 2000, 3000]
#    bio_x_dict: {"prs": [[1.0, -1.0, 0.5]], "wbc": [[0.0, 2.0, 0.5]]}
#    bio_t: [0, 2500]
#    bio_m: [2, 3]
# ShapArray:
#    all_x: [2, 101, 1269, 1.0, -1.0, 0.5, 0.0, 2.0, 0.5]
#    all_t: [0, 2000, 3000, 0, 0, 0, 2500, 2500, 2500]
#    all_m: [1, 1, 1, 2, 2, 2, 3, 3, 3]


def to_shap_array(
    s: MultimodalOut,
    detokenizer: dict,
    biomarker_features: dict[Modality, list],
    biomarker_background: dict[Modality, np.ndarray],
) -> tuple[ShapArray, list[str], np.ndarray]:

    all_x, all_t, all_m = list(), list(), list()
    x, t, bio_x_dict, bio_t, bio_m = s
    all_x.append(x.astype(np.float32))
    all_t.append(t)
    all_m.append(np.ones_like(x))

    features = list()
    token_features = [detokenizer[int(_x)] for _x in x]
    features.extend(token_features)

    bio_bg = list()
    # ordering ensures reproducibility
    _, _idx = np.unique(bio_m, return_index=True)
    modvals = bio_m[np.sort(_idx)]
    for modval in modvals:
        modality = Modality(modval)
        for _bio_x, _bio_t in zip(bio_x_dict[modality], bio_t[bio_m == modval]):
            #! for each modality, assume no more than 1 measurement at each timestep
            all_x.append(_bio_x)
            all_t.append(np.full_like(_bio_x, fill_value=_bio_t))
            all_m.append(np.full_like(_bio_x, fill_value=modval))
            _features = [f"{modality.name}.{f}" for f in biomarker_features[modality]]
            features.extend(_features)
            bio_bg.append(biomarker_background[modality])
    all_x = np.concatenate(all_x)
    all_t = np.concatenate(all_t)
    all_m = np.concatenate(all_m)
    if len(bio_bg) > 0:
        bio_bg = np.concatenate(bio_bg)
    else:
        bio_bg = np.array([])

    return (all_x, all_t, all_m), features, bio_bg


def from_shap_array(
    s: ShapArray,
    biomarker_features: dict[Modality, list],
) -> MultimodalOut:

    all_x, all_t, all_m = s
    token_mask = all_m == 1
    x, t = all_x[token_mask].astype(np.int64), all_t[token_mask].astype(np.float32)

    bio_x_dict, bio_t, bio_m = dict(), list(), list()

    _, _idx = np.unique(all_m, return_index=True)
    modvals = all_m[np.sort(_idx)]

    for modval in modvals:
        if modval <= 1:
            continue
        mod_mask = all_m == modval
        mod_x, mod_t = all_x[mod_mask], all_t[mod_mask]
        modality = Modality(modval)
        bio_x_dict[modality] = list()

        # Use known feature dimension to correctly split measurements
        feature_dim = len(biomarker_features[modality])
        n_measurements = len(mod_x) // feature_dim
        for i in range(n_measurements):
            start, end = i * feature_dim, (i + 1) * feature_dim
            bio_x_dict[modality].append(mod_x[start:end])
            bio_t.append(mod_t[start])  # All timestamps in chunk are identical
            bio_m.append(modval)
    bio_t, bio_m = np.array(bio_t).astype(np.float32), np.array(bio_m).astype(np.int64)
    bio_t, bio_m = sort_by_time(bio_t, bio_m)

    return x, t, bio_x_dict, bio_t, bio_m


class MultimodalShapMasker(shap.maskers.Masker):  # type: ignore

    def __init__(
        self,
        biomarker_background: np.ndarray,
        biomarker_only: bool = False,
    ):
        self.base_masker = ShapMasker()
        self.biomarker_background = biomarker_background
        self.biomarker_only = biomarker_only

        self.immutable_outputs = True
        self.fixed_background = False

    def shape(self, s: ShapArray):
        if self.biomarker_only:
            all_x, all_t, all_m = s
            return (1, int((all_m != 1).sum()))
        return (1, len(s[0]))

    def mask_shapes(self, s: ShapArray):
        if self.biomarker_only:
            all_x, all_t, all_m = s
            return [(int((all_m != 1).sum()),)]
        return [(len(s[0]),)]

    def __call__(self, mask, s: ShapArray):
        mask = self._standardize_mask(mask, s)
        all_x, all_t, all_m = s
        all_x, all_t, all_m = all_x.copy(), all_t.copy(), all_m.copy()

        is_tok = all_m == 1
        x, t = all_x[is_tok], all_t[is_tok]
        all_bio_x = all_x[~is_tok]
        all_bio_t = all_t[~is_tok]
        all_bio_m = all_m[~is_tok]

        if self.biomarker_only:
            bio_mask = mask  # mask is already sized for biomarkers only
        else:
            tok_mask = mask[is_tok]
            ((x,), (t,)) = self.base_masker(mask=tok_mask, s=(x, t))
            bio_mask = mask[~is_tok]

        all_bio_x[~bio_mask] = self.biomarker_background[~bio_mask]
        all_x = np.concatenate([x, all_bio_x])
        all_t = np.concatenate([t, all_bio_t])
        all_m = np.concatenate([np.ones_like(x, dtype=all_m.dtype), all_bio_m])

        return (
            (all_x,),
            (all_t,),
            (all_m,),
        )


@torch.no_grad
def multimodal_shap_forward(
    all_x_lst: list[np.ndarray],
    all_t_lst: list[np.ndarray],
    all_m_lst: list[np.ndarray],
    biomarker_features: dict[Modality, list],
    model,
):
    x_lst, t_lst = list(), list()
    biomarker, bio_t_lst, bio_m_lst = defaultdict(list), list(), list()

    for all_x, all_t, all_m in zip(all_x_lst, all_t_lst, all_m_lst):
        x, t, bio_x_dict, bio_t, bio_m = from_shap_array(
            (all_x, all_t, all_m), biomarker_features=biomarker_features
        )
        x_lst.append(x)
        t_lst.append(t)
        for modality in bio_x_dict.keys():
            biomarker[modality].extend(bio_x_dict[modality])
        bio_t_lst.append(bio_t)
        bio_m_lst.append(bio_m)

    device = next(model.parameters()).device
    idx = collate_batch(x_lst)
    idx = torch.tensor(idx, dtype=torch.long).to(device)
    age = collate_batch(t_lst, fill_value=-1e4)
    age = torch.tensor(age, dtype=torch.float32).to(device)
    for modality, bio_x_lst in biomarker.items():
        biomarker[modality] = torch.from_numpy(np.stack(bio_x_lst)).to(device)  # type: ignore
    mod_age = collate_batch(bio_t_lst, fill_value=-1e4)
    mod_age = torch.tensor(mod_age, dtype=torch.float32).to(device)
    mod_idx = collate_batch(bio_m_lst)
    mod_idx = torch.tensor(mod_idx, dtype=torch.long).to(device)

    outputs, _, _ = model.forward(idx, age, biomarker, mod_age, mod_idx)
    logits = outputs["logits"][:, -1, :]
    logits[:, model.config.ignore_tokens] = -torch.inf

    return logits.detach().cpu().numpy()
