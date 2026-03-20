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

    def shape(self, dummy: np.ndarray):
        return (1, len(dummy))

    def mask_shapes(self, dummy: np.ndarray):
        return [(len(dummy),)]

    def __call__(self, mask, dummy: np.ndarray):
        mask = self._standardize_mask(mask, dummy)
        return ((mask.astype(np.int8),),)


class MultimodalShapModel:

    def __init__(
        self,
        model: torch.nn.Module,
        biomarker_only: bool,
        biomarker_background: None | dict,
        biomarker_features: dict,
        data: MultimodalOut,
    ):
        self.model = model
        self.biomarker_only = biomarker_only
        self.biomarker_background = biomarker_background
        self.biomarker_features = biomarker_features
        self.x, self.t, self.bio_x_dict, self.bio_t, self.bio_m = data

        self.n_tokens = len(self.t)
        if self.biomarker_background is None:
            self.features = [Modality(int(mval)).name for mval in self.bio_m]
            self.timesteps = self.bio_t
            self.mask_size = self.bio_m.size
        else:
            self.features = list()
            self.timesteps = []
            self.mask_size = 0
            for i, modval in enumerate(self.bio_m):
                mod = Modality(int(modval))
                mod_features = [f"{mod.name}.{k}" for k in biomarker_features[mod]]
                self.features.extend(mod_features)
                n_mod_features = len(mod_features)
                self.timesteps.append(np.full(n_mod_features, self.bio_t[i]).tolist())
                self.mask_size += n_mod_features
            self.timesteps = np.concatenate(self.timesteps)

        self.n_biomarker_features = self.mask_size
        if not self.biomarker_only:
            self.mask_size += self.n_tokens
            self.timesteps = np.concatenate([self.t, self.timesteps])
            self.features = self.x.tolist() + self.features

    def dummy(self):
        return np.ones(self.mask_size)

    def mask_biomarkers_with_missing(self, mask: np.ndarray):
        mask_bool = mask.astype(bool)
        bio_t = self.bio_t[mask_bool]
        bio_m = self.bio_m[mask_bool]

        # Apply mask to bio_x_dict.
        # bio_x_dict[M][k] is the k-th measurement of modality M in chronological
        # order, matching the k-th occurrence of M in bio_m. We track per-modality
        # counters to select the right rows.
        per_mod_counter: dict = defaultdict(int)
        bio_x_dict: dict = defaultdict(list)
        for present, mval in zip(mask_bool, self.bio_m):
            mod = Modality(int(mval))
            k = per_mod_counter[mod]

            if present:
                bio_x_dict[mod].append(self.bio_x_dict[mod][k])

            per_mod_counter[mod] += 1

        return bio_x_dict, bio_t, bio_m

    def mask_biomarkers_with_background(self, mask: np.ndarray):
        mask_bool = mask.astype(bool)
        offset = 0
        per_mod_counter: dict = defaultdict(int)
        bio_x_dict: dict = defaultdict(list)
        for mval in self.bio_m:
            mod = Modality(int(mval))
            k = per_mod_counter[mod]
            n_features = len(self.biomarker_features[mod])
            bio_mask = mask_bool[offset : offset + n_features]

            bio_x = self.biomarker_background[mod].copy()  # type: ignore
            bio_x[bio_mask] = self.bio_x_dict[mod][k][bio_mask]
            bio_x_dict[mod].append(bio_x)

            offset += n_features
            per_mod_counter[mod] += 1

        return bio_x_dict, self.bio_t, self.bio_m

    def mask_tokens(self, mask: np.ndarray):
        x, t = self.x.copy(), self.t.copy()
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

        return x, t

    @torch.no_grad
    def __call__(self, masks: list[np.ndarray]):
        x_lst, t_lst = [], []
        batched_bio_x: dict = defaultdict(list)
        bio_t_lst, bio_m_lst = [], []

        for mask in masks:

            if not self.biomarker_only:
                x, t = self.mask_tokens(mask[: self.n_tokens])
                x_lst.append(x)
                t_lst.append(t)
            else:
                x_lst.append(self.x)
                t_lst.append(self.t)

            if self.biomarker_background is not None:
                bio_x, bio_t, bio_m = self.mask_biomarkers_with_background(
                    mask[-self.n_biomarker_features :]
                )
            else:
                bio_x, bio_t, bio_m = self.mask_biomarkers_with_missing(
                    mask[-self.n_biomarker_features :]
                )

            for mod, arrays in bio_x.items():
                batched_bio_x[mod].extend(arrays)
            bio_t_lst.append(bio_t)
            bio_m_lst.append(bio_m)

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        idx = collate_batch(x_lst)
        idx = torch.tensor(idx, dtype=torch.long).to(device)
        age = collate_batch(t_lst, fill_value=-1e4)
        age = torch.tensor(age, dtype=dtype).to(device)

        bio_x_batch = {}
        for mod, arrays in batched_bio_x.items():
            bio_x_batch[mod] = torch.from_numpy(np.stack(arrays)).to(
                dtype=dtype, device=device
            )

        mod_age = collate_batch(bio_t_lst, fill_value=-1e4)
        mod_age = torch.tensor(mod_age, dtype=dtype).to(device)
        mod_idx = collate_batch(bio_m_lst)
        mod_idx = torch.tensor(mod_idx, dtype=torch.long).to(device)

        outputs, _, _ = self.model.forward(idx, age, bio_x_batch, mod_age, mod_idx)
        logits = outputs["logits"][:, -1, :]
        logits[:, self.model.config.ignore_tokens] = -torch.inf

        return logits.detach().cpu().numpy()
