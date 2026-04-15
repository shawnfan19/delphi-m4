import functools
from typing import Literal

import numpy as np

from delphi.data.utils import (
    append_no_event,
    crop_contiguous,
    dissolve_clusters,
    exclude_tokens,
    identity_transform,
    perturb_time,
    sort_by_time,
)


class TokenTransform:

    def __init__(
        self,
        no_event_interval: None | float = 5 * 365.25,
        no_event_mode: str = "legacy-random",
        block_size: None | int = None,
        perturb_tokens: None | list = None,
        blacklist_tokens: None | list = None,
        crop_mode: Literal["left", "right", "random"] = "right",
        deterministic: bool = False,
        seed: int = 42,
        break_clusters: bool = False,
        dx_token: None | int = None,
        whitelist_tokens: list | np.ndarray | None = None,
    ):

        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.deterministic = deterministic

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
            self.break_clusters = functools.partial(
                dissolve_clusters,
                whitelist=np.array(whitelist_tokens),
                dx_token=self.dx_token,
            )
        else:
            self.break_clusters = identity_transform

    def __call__(self, x_pid, t_pid):

        if self.deterministic:
            rng = np.random.default_rng(int(x_pid.sum()) + self.seed)
        else:
            rng = self.rng

        x_pid, t_pid = self.exclude_tokens(x_pid, t_pid)
        x_pid, t_pid = self.append_no_event(x_pid, t_pid, rng=rng)
        x_pid, t_pid = self.perturb_time(x_pid, t_pid, rng=rng)
        t_pid, x_pid = sort_by_time(t_pid, x_pid, stable=self.deterministic)
        x_pid, t_pid = self.crop_block_size(x_pid, t_pid, rng=rng)
        x_pid, t_pid = self.break_clusters(x_pid, t_pid, rng=rng)

        return x_pid, t_pid
