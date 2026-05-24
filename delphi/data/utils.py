from typing import Literal

import numpy as np


def collate_batch(
    batch_data: list[np.ndarray], fill_value: int | float = 0, pad_left: bool = True
) -> np.ndarray:

    max_len = max([bd.size for bd in batch_data])
    collated_batch = np.full(
        shape=(len(batch_data), max_len),
        fill_value=fill_value,
        dtype=batch_data[0].dtype,
    )
    for i, bd in enumerate(batch_data):
        if bd.size > 0:
            if pad_left:
                collated_batch[i, -bd.size :] = bd
            else:
                collated_batch[i, : bd.size] = bd

    return collated_batch


def collate_batches(
    batch_data: list[np.ndarray], fill_value: int | float = 0, pad_left: bool = True
) -> np.ndarray:

    max_len = max([bd.shape[1] for bd in batch_data])
    n_lst = np.array([bd.shape[0] for bd in batch_data])
    collated_batch = np.full(
        shape=(n_lst.sum(), max_len),
        fill_value=fill_value,
        dtype=batch_data[0].dtype,
    )

    s = 0
    for i, bd in enumerate(batch_data):
        l = bd.shape[0]
        if pad_left:
            collated_batch[s : s + l, -bd.shape[1] :] = bd
        else:
            collated_batch[s : s + l, : bd.shape[1]] = bd
        s += l

    return collated_batch


def identity_transform(*args, **kwargs):
    return args


def forward_fill(dt, mask):
    idx = np.arange(len(dt))
    val_idx = np.where(~mask, idx, 0)
    ffill_idx = np.maximum.accumulate(val_idx)
    return dt[ffill_idx]


def dissolve_clusters(
    x: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    whitelist: np.ndarray,
    dx_token: int,
):
    t = t.copy()
    x = x.copy()

    is_dis = ~np.isin(x, whitelist)
    uniq_t = np.unique(t[is_dis])
    diag_t = uniq_t
    diag_x = np.full_like(diag_t, fill_value=dx_token)

    dt = np.diff(t)
    dt = np.insert(dt, 0, t[0])
    dt = forward_fill(dt, dt == 0)
    assert dt.min() >= 0

    perturb_t = rng.uniform(size=len(dt))
    perturb_t *= dt
    t[is_dis] = t[is_dis] - perturb_t[is_dis]

    t = np.concatenate((t, diag_t))
    x = np.concatenate((x, diag_x))

    t, x = sort_by_time(t, x)

    return x.astype(np.uint32), t.astype(np.float32)


def pack_clusters(tokens, timesteps, whitelist, dx_token=1):
    batch_size = tokens.shape[0]
    is_dx_token = tokens == dx_token
    if dx_token == 1:
        prev_token = np.concatenate(
            (np.full((batch_size, 1), fill_value=0), tokens[:, :-1]), axis=1
        )
        is_dx_token = np.logical_and(is_dx_token, ~np.isin(prev_token, whitelist))
    is_whitelist = np.isin(tokens, whitelist)
    to_pack = np.logical_and(~is_whitelist, ~is_dx_token)
    timesteps = backward_fill(timesteps, to_pack, axis=1)

    tokens[is_dx_token] = 0
    timesteps[is_dx_token] = -1e4

    sort_by_age = np.argsort(timesteps, axis=1)
    timesteps = np.take_along_axis(timesteps, sort_by_age, axis=1)
    tokens = np.take_along_axis(tokens, sort_by_age, axis=1)

    return tokens, timesteps


def backward_fill(t, mask, axis=-1):
    """
    Args:
        t: Data array (e.g. shape [Batch, Time])
        mask: Boolean mask, True indicates missing value
        axis: The axis to fill along (default -1)
    """
    idx_len = t.shape[axis]

    # 1. Create indices [0, 1, ... L-1]
    # We reshape it so it broadcasts against t (e.g. shape [1, L] for 2D)
    idx = np.arange(idx_len)
    shape_view = [1] * t.ndim
    shape_view[axis] = idx_len
    idx = idx.reshape(shape_view)

    # 2. Fill masked areas with the LAST index (L-1)
    # This prepares the array for minimum accumulation from right-to-left
    val_idx = np.where(~mask, idx, idx_len - 1)

    # 3. Propagate indices backwards
    # NumPy accumulate works left-to-right, so we:
    # Flip -> Accumulate Minimum -> Flip Back
    val_idx_flipped = np.flip(val_idx, axis=axis)
    bfill_idx_flipped = np.minimum.accumulate(val_idx_flipped, axis=axis)
    bfill_idx = np.flip(bfill_idx_flipped, axis=axis)

    # 4. Use take_along_axis to fetch values
    # t[bfill_idx] would not work correctly in 2D+
    return np.take_along_axis(t, bfill_idx, axis=axis)


def append_no_event(
    x: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    interval: float,
    mode: str = "random",
    token: int = 1,
) -> tuple[np.ndarray, np.ndarray]:

    if mode == "random":
        max_age = np.max(t)
        # add 1e-6 to ensure no_event does not co-occur with first token
        min_age = max(np.min(t[t >= 0]), 0) + 1e-6
        age_range = max_age - min_age
        n = int(age_range // interval) - 1
        if n <= 0:
            no_event_t = np.array([])
        else:
            no_event_t = rng.uniform(min_age, max_age, size=(n,))
    elif mode == "regular":
        max_age = np.max(t)
        min_age = max(np.min(t[t >= 0]), 0) + 1e-6
        age_range = max_age - min_age
        n = int(age_range // interval) - 1
        if n <= 0:
            no_event_t = np.array([])
        else:
            no_event_t = np.linspace(min_age, max_age, num=n)
    elif mode == "legacy-random":
        min_age = np.min(t[t >= 0])
        max_age = np.max(t)
        no_event_t = rng.uniform(1, 36525, size=(int(36525 / interval),))
        no_event_t = no_event_t[
            np.logical_and(no_event_t >= min_age, no_event_t < max_age)
        ]
    elif mode == "exponential":
        rate = 1 / interval
        dt = np.diff(t)
        n_gaps = len(dt)
        max_per_gap = int(rate * dt.max() * 4)
        exp_samples = rng.exponential(1 / rate, size=(n_gaps, max_per_gap))
        cumsum = np.cumsum(exp_samples, axis=1)
        valid_mask = cumsum < dt[:, None]
        absolute_times = t[:-1, None] + cumsum
        no_event_t = absolute_times[valid_mask]
    else:
        raise ValueError

    no_event_t = no_event_t.astype(np.float32)
    no_event_x = np.full(no_event_t.shape, token)

    x = np.concatenate((x, no_event_x))
    t = np.concatenate((t, no_event_t))

    return x, t


def move_to_last(x: np.ndarray, t: np.ndarray, token: int):

    n = (x == token).sum()
    assert n < 2
    if n > 0:
        idx = np.zeros_like(x)
        idx[x == token] = 1
        move_last = np.argsort(x, stable=True)
        x = x[move_last]
        t = t[move_last]

    return x, t


def _crop_slice(mode, max_len, block_size, rng):
    if mode == "left":
        start = 0
    elif mode == "right":
        start = max_len - block_size
    elif mode == "random":
        start = rng.integers(0, max_len - block_size + 1)
    else:
        raise ValueError
    return slice(start, start + block_size)


def crop_contiguous(
    x: np.ndarray,
    *args: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
    mode: Literal["left", "right", "random"] = "left",
):
    """
    input sequences should be sorted according to time
    """

    L = x.shape[0]
    if L <= block_size:
        return (x, *args) if args else x
    else:
        cut = _crop_slice(mode, L, block_size, rng)
        if args:
            return x[cut], *[arr[cut] for arr in args]
        else:
            return x[cut]


def filter_biomarker_array(
    bio_x_dict, bio_t, bio_m, mask: np.ndarray, biomarker2idx: dict[str, int]
):
    """Boolean mask over t/m, propagated per-modality into x.

    bio_x_dict is keyed by lowercase biomarker name; biomarker2idx maps that
    name to the integer used in bio_m.
    """
    new_bio_x_dict = {}
    for mod, vals in bio_x_dict.items():
        mod_mask = mask[bio_m == biomarker2idx[mod]]
        filtered = [v for v, k in zip(vals, mod_mask) if k]
        if filtered:
            new_bio_x_dict[mod] = filtered
    return new_bio_x_dict, bio_t[mask], bio_m[mask]


def dropout_biomarkers(
    bio_x_dict: dict,
    bio_t: np.ndarray,
    bio_m: np.ndarray,
    rng: np.random.Generator,
    p: float,
    biomarker2idx: dict[str, int],
) -> tuple[dict, np.ndarray, np.ndarray]:
    if len(bio_t) == 0:
        return bio_x_dict, bio_t, bio_m

    keep = rng.random(size=len(bio_t)) >= p

    return filter_biomarker_array(
        bio_x_dict, bio_t, bio_m, mask=keep, biomarker2idx=biomarker2idx
    )


def remove_after_np(x, t, bio_x_dict, bio_t, bio_m, cutoff_t, biomarker2idx):
    """Remove tokens and biomarker measurements after cutoff_t.

    Unbatched version for __getitem__ outputs: x, t are 1D numpy arrays;
    bio_x_dict values are lists of numpy arrays; bio_t, bio_m are 1D numpy.
    """
    x_mask = t <= cutoff_t
    x = x[x_mask]
    t = t[x_mask]

    bio_x_dict, bio_t, bio_m = filter_biomarker_array(
        bio_x_dict, bio_t, bio_m, mask=bio_t <= cutoff_t, biomarker2idx=biomarker2idx
    )

    return x, t, bio_x_dict, bio_t, bio_m


def sort_by_time(t: np.ndarray, *args: np.ndarray, stable: bool = False):
    s = np.argsort(t, stable=stable)
    t = t[s]
    return t, *[arg[s] for arg in args]


def perturb_time(
    x: np.ndarray,
    t: np.ndarray,
    tokens: np.ndarray,
    rng: np.random.Generator,
    low: float = -20 * 365.25,
    high: float = 40 * 365.25,
):
    to_perturb = np.isin(x, tokens)
    t[to_perturb] += rng.uniform(low=low, high=high, size=(to_perturb.sum(),))
    return x, t


def exclude_tokens(x: np.ndarray, t: np.ndarray, blacklist: np.ndarray):
    to_exclude = np.isin(x, blacklist)
    x = x[~to_exclude]
    t = t[~to_exclude]
    return x, t


def remove_after(x, t, bio_x_dict, bio_t, bio_m, cutoff_t, biomarker2idx):
    """Remove tokens and biomarker measurements after cutoff_t.

    Batched version: x, t are 2D tensors; bio_t, bio_m are 2D tensors.
    bio_x_dict is keyed by lowercase biomarker name; biomarker2idx maps that
    name to the integer used in bio_m.
    """
    x_mask = t.ravel() <= cutoff_t
    x = x[:, x_mask].clone()
    t = t[:, x_mask].clone()

    bio_mask = bio_t.ravel() <= cutoff_t
    bio_x_dict = {
        mod: v[bio_mask[bio_m.ravel() == biomarker2idx[mod]]].clone()
        for mod, v in bio_x_dict.items()
    }
    bio_t = bio_t[:, bio_mask].clone()
    bio_m = bio_m[:, bio_mask].clone()

    return x, t, bio_x_dict, bio_t, bio_m


def update_tokenizer(base_tokenizer: dict, add_tokenizer: dict) -> tuple[dict, int]:

    assert min(base_tokenizer.values()) == 0, "base tokenizer must start with 0"
    assert min(add_tokenizer.values()) == 1, "additional tokenizer must start with 1"
    offset = len(base_tokenizer) - 1
    for key, value in add_tokenizer.items():
        if key not in base_tokenizer:
            base_tokenizer[key] = value + offset
        else:
            raise ValueError(f"{key} already exists in base tokenizer")
    return base_tokenizer, offset
