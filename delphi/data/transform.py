import functools
import pprint
from typing import Literal

import numpy as np

from delphi.data.utils import (
    append_no_event,
    crop_contiguous,
    dissolve_clusters,
    dropout_biomarkers,
    exclude_tokens,
    filter_biomarker_array,
    identity_transform,
    perturb_time,
    sort_by_time,
    group_by_time_to_multihot,
)

SET_SEQUENCE_MODES = {
    "dynamic_bernoulli",
    "interval_dynamic_bernoulli",
}

class TokenTransform:

    def __init__(
        self,
        no_event_interval: None | float = 5 * 365.25,
        no_event_mode: str = "legacy-random",
        block_size: None | int = None,
        perturb_tokens: None | list = None,
        blacklist_tokens: None | list = None,
        crop_mode: Literal["left", "right", "random"] = "right",
        deterministic: bool = True,
        seed: int = 42,
        break_clusters: bool = False,
        dx_token: None | int = None,
        whitelist_tokens: list | np.ndarray | None = None,
        death_token: None | int = None,
        ignore_tokens: list | np.ndarray | None = None,
        epsilon: float = 0.01,
        set_mode: Literal["none", "dynamic_bernoulli", "interval_dynamic_bernoulli"] = "none",
        vocab_size: None | int = None,
    ):
        """
        args:
            no_event_interval: average time intervals for introducing no-event tokens
            no_event_mode: mode for introducing no-event tokens
                refer to append_no_event for more details
            perturb_tokens: a list of tokens whose timestamps are perturbed for data augmentation
                default is the lifestyle tokens in the UKB
            block_size: maximum sequence length
            crop_mode: where to start cropping a sequence that exceeds block_size
                "left": start from the beginning
                "right": start from the end
                "random": start from a random position in the middle
            seed: random seed for reproducibility
            deterministic: if True, the same participant will always receive the same augmentations.
        """

        self._init_args = {k: v for k, v in locals().items() if k != "self"}

        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.deterministic = deterministic

        self.set_mode = "none" if set_mode is None else set_mode
        if self.set_mode not in ({"none"} | SET_SEQUENCE_MODES):
            raise ValueError(
                f"Unknown set_mode={self.set_mode!r}. "
                f"Expected one of {sorted({'none'} | SET_SEQUENCE_MODES)}"
            )

        self.return_sets = self.set_mode in SET_SEQUENCE_MODES
        self.vocab_size = vocab_size

        if self.return_sets:
            if vocab_size is None:
                raise ValueError(
                    "TokenTransform set modes require vocab_size so token ids can "
                    "be grouped into multi-hot set vectors."
                )
            if break_clusters:
                raise ValueError(
                    "Set-valued Dynamic Bernoulli modes require break_clusters=False. "
                    "Tied timestamps are grouped into one multi-hot set event instead."
                )
            if dx_token is not None:
                raise ValueError(
                    "Set-valued Dynamic Bernoulli modes use no dx_token."
                )

        self.no_event_interval = no_event_interval
        if no_event_interval is not None:
            self.append_no_event = functools.partial(
                append_no_event,
                interval=no_event_interval,
                token=1,
                mode=no_event_mode,
            )
        else:
            self.append_no_event = identity_transform

        if blacklist_tokens is not None:
            self.exclude_tokens = functools.partial(
                exclude_tokens, blacklist=np.array(blacklist_tokens)
            )
        else:
            self.exclude_tokens = identity_transform

        if perturb_tokens is not None:
            self.perturb_time = functools.partial(
                perturb_time,
                tokens=np.array(perturb_tokens),
            )
        else:
            self.perturb_time = identity_transform

        if block_size is not None:
            self.crop_block_size = functools.partial(
                crop_contiguous,
                block_size=block_size,
                mode=crop_mode,
            )
        else:
            self.crop_block_size = identity_transform

        self.dx_token = None
        if break_clusters:
            assert dx_token is not None
            self.dx_token = dx_token

            assert whitelist_tokens is not None
            assert death_token is not None
            self.break_clusters = functools.partial(
                dissolve_clusters,
                whitelist=np.array(whitelist_tokens),
                dx_token=self.dx_token,
                death_token=death_token,
                ignore_tokens=np.array(
                    ignore_tokens if ignore_tokens is not None else [], dtype=int
                ),
                epsilon=epsilon,
            )
        else:
            self.break_clusters = identity_transform

    @property
    def config(self) -> dict:
        return dict(self._init_args)

    @classmethod
    def from_ckpt(cls, ckpt_dict: dict) -> "TokenTransform":
        return cls(**ckpt_dict["token_transform_args"])

    def to_ckpt(self) -> dict:
        return {"token_transform_args": self.config}

    def replace(self, **overrides) -> "TokenTransform":
        return type(self)(**{**self._init_args, **overrides})

    def describe(self) -> None:
        print(f"{type(self).__name__}:")
        heavy_keys = {"perturb_tokens", "blacklist_tokens", "whitelist_tokens"}
        display = {}
        for k, v in self.config.items():
            if k in heavy_keys and v is not None:
                arr = np.asarray(v)
                display[k] = f"<array shape={arr.shape}>"
            else:
                display[k] = v
        pprint.pp(display)

    def __call__(self, x_pid, t_pid):

        if self.deterministic:
            rng = np.random.default_rng(int(x_pid.sum()) + self.seed)
        else:
            rng = self.rng

        x_pid, t_pid = self.exclude_tokens(x_pid, t_pid)
        x_pid, t_pid = self.append_no_event(x_pid, t_pid, rng=rng)
        x_pid, t_pid = self.perturb_time(x_pid, t_pid, rng=rng)
        t_pid, x_pid = sort_by_time(t_pid, x_pid, stable=self.deterministic)

        if self.return_sets:
            # Group all tokens with exactly identical timestamps into one set event.
            # This is the set-valued replacement for dissolve_clusters + dx_token.
            x_pid, t_pid = group_by_time_to_multihot(
                x=x_pid,
                t=t_pid,
                vocab_size=int(self.vocab_size),
            )

            # In set modes, block_size counts set-events, not individual tokens.
            x_pid, t_pid = self.crop_block_size(x_pid, t_pid, rng=rng)

            return x_pid, t_pid

        # Old token-level path.
        x_pid, t_pid = self.crop_block_size(x_pid, t_pid, rng=rng)
        x_pid, t_pid = self.break_clusters(x_pid, t_pid, rng=rng)

        return x_pid, t_pid


class BiomarkerTransform:

    def __init__(
        self,
        biomarker2idx: dict[str, int],
        first_time_only: bool = False,
        dropout: None | float = None,
        z_score: bool = False,
        mean: None | dict[str, np.ndarray] = None,
        std: None | dict[str, np.ndarray] = None,
        deterministic: bool = True,
        seed: int = 42,
    ):
        """
        args:
            biomarker2idx: mapping from lowercase biomarker name to the integer
                used in bio_m. Source of truth lives on the reader.
            first_time_only: if True only use the first occurrence of each biomarker
            dropout: if not None, each biomarker measurement is independently dropped with this probability
            z_score: whether to z-score biomarker values
        """
        self._init_args = {k: v for k, v in locals().items() if k != "self"}

        self.biomarker2idx = biomarker2idx
        self.first_time_only = first_time_only

        if dropout is not None:
            self.dropout_biomarkers = functools.partial(
                dropout_biomarkers,
                p=dropout,
                biomarker2idx=biomarker2idx,
            )
        else:
            self.dropout_biomarkers = identity_transform

        if z_score:
            assert mean is not None
            assert std is not None
        self.z_score = z_score
        self.mean = mean
        self.std = std
        self.seed = seed
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    @property
    def config(self) -> dict:
        # biomarker2idx lives on the model config; don't duplicate it here
        return {
            k: v
            for k, v in self._init_args.items()
            if k not in {"mean", "std", "biomarker2idx"}
        }

    @classmethod
    def from_ckpt(cls, ckpt_dict: dict) -> "BiomarkerTransform | None":
        args = ckpt_dict.get("biomarker_transform_args")
        if not args:
            return None
        stats = ckpt_dict.get("biomarker_stats") or {}
        biomarker2idx = ckpt_dict["model_args"]["biomarker2idx"]
        return cls(
            biomarker2idx=biomarker2idx,
            **args,
            mean=stats.get("mean"),
            std=stats.get("std"),
        )

    def to_ckpt(self) -> dict:
        return {
            "biomarker_transform_args": self.config,
            "biomarker_stats": self.stats,
        }

    def replace(self, **overrides) -> "BiomarkerTransform":
        return type(self)(**{**self._init_args, **overrides})

    @property
    def stats(self) -> dict:
        def keys_to_str(d):
            if d is None:
                return None
            return {k.lower(): v for k, v in d.items()}

        return {"mean": keys_to_str(self.mean), "std": keys_to_str(self.std)}

    def describe(self) -> None:
        print(f"{type(self).__name__}:")
        pprint.pp(self.config)

        mean = self.mean or {}
        std = self.std or {}
        biomarkers = sorted(set(mean) | set(std))
        if not biomarkers:
            return
        width = max(len(m) for m in biomarkers)
        print("stats:")
        with np.printoptions(precision=4, linewidth=120):
            for m in biomarkers:
                mu = np.asarray(mean.get(m)) if m in mean else None
                sigma = np.asarray(std.get(m)) if m in std else None
                shape = mu.shape if mu is not None else sigma.shape
                print(f"  {m:<{width}}  shape={shape}")
                print(f"    mean: {mu}")
                print(f"    std:  {sigma}")

    def __call__(self, bio_x_dict, bio_t, bio_m):

        if self.deterministic:
            rng = np.random.default_rng(int(bio_m.sum()) + self.seed)
        else:
            rng = self.rng

        if self.first_time_only:
            seen = set()
            keep = np.zeros(len(bio_m), dtype=bool)
            for i, m in enumerate(bio_m):
                if m not in seen:
                    seen.add(m)
                    keep[i] = True
            bio_x_dict, bio_t, bio_m = filter_biomarker_array(
                bio_x_dict, bio_t, bio_m, mask=keep, biomarker2idx=self.biomarker2idx
            )

        if self.z_score:
            bio_x_dict = {
                mod: [(x - self.mean[mod]) / self.std[mod] for x in vals]
                for mod, vals in bio_x_dict.items()
            }

        bio_x_dict, bio_t, bio_m = self.dropout_biomarkers(
            bio_x_dict, bio_t, bio_m, rng=rng
        )

        return bio_x_dict, bio_t, bio_m

    def untransform(self, bio_x_dict):
        """Map z-scored biomarker values back to raw units (inverse of __call__'s
        z-score, ``x * std + mean``).

        Accepts ``{biomarker -> array}`` with features on the last axis — e.g. a
        single measurement vector, or the per-biomarker matrices returned by
        BiomarkerCollator.finalize. Identity when ``z_score`` is off. Only the
        z-score is inverted; dropout / first_time_only are not undone. NaNs (e.g.
        absent participants) pass through unchanged.
        """
        if not self.z_score:
            return bio_x_dict
        return {
            mod: x * self.std[mod] + self.mean[mod] for mod, x in bio_x_dict.items()
        }


class MultimodalPrompt:
    """instance-level prompt cutting for multimodal data.

    splits a full sequence into prompt and ground truth, and removes
    biomarker measurements after the prompt cutoff. `prompt_age` may be a
    scalar shared across participants, or a dict mapping pid -> age.
    """

    def __init__(
        self,
        prompt_age: float | dict,
        biomarker2idx: dict[str, int],
        append_no_event: bool = False,
    ):
        self._init_args = {k: v for k, v in locals().items() if k != "self"}

        self.prompt_age = prompt_age
        self.biomarker2idx = biomarker2idx
        self.append_no_event = append_no_event

    def __call__(self, x, t, bio_x_dict, bio_t, bio_m, pid=None):
        if isinstance(self.prompt_age, dict):
            assert pid is not None, "pid is required when prompt_age is a mapping"
            cutoff = self.prompt_age[pid]
        else:
            cutoff = self.prompt_age

        # prompt tokens: everything up to cutoff
        pmt_mask = t <= cutoff
        pmt_x = x[pmt_mask]
        pmt_t = t[pmt_mask]

        if self.append_no_event:
            pmt_x = np.append(pmt_x, 1)
            pmt_t = np.append(pmt_t, cutoff)

        # filter biomarkers to prompt window
        bio_keep = bio_t <= cutoff
        bio_x_dict, bio_t, bio_m = filter_biomarker_array(
            bio_x_dict, bio_t, bio_m, mask=bio_keep, biomarker2idx=self.biomarker2idx
        )

        return pmt_x, pmt_t, bio_x_dict, bio_t, bio_m, x, t
