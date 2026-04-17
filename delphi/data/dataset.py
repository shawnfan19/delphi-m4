from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
import torch

from delphi.data.transform import TokenTransform
from delphi.data.ukb import UKBReader
from delphi.data.utils import (
    collate_batch,
)
from delphi.env import DELPHI_DATA_READ as DELPHI_DATA_DIR


class Dataset:

    def __init__(
        self,
        subject_list: str = "participants/train_fold.bin",
        no_event_interval: None | float = 5 * 365.25,
        no_event_mode: str = "legacy-random",
        block_size: None | int = None,
        perturb_list: None | list = None,
        exclude_list: None | list = None,
        crop_mode: Literal["left", "right", "random"] = "right",
        break_clusters: bool = False,
        additional_dx_token: bool = True,
        seed: int = 42,
        deterministic: bool = False,
    ):

        self._init_args = locals().copy()
        self._init_args.pop("self")  # Remove 'self' reference

        self.reader = UKBReader()
        self.tokenizer = self.reader.tokenizer

        participants_path = Path(DELPHI_DATA_DIR) / "ukb_real_data" / subject_list
        self.participants = np.fromfile(participants_path, dtype=np.uint32)

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic

        self.token_transform = TokenTransform(
            no_event_interval=no_event_interval,
            no_event_mode=no_event_mode,
            block_size=block_size,
            perturb_tokens=perturb_list,
            blacklist_tokens=exclude_list,
            crop_mode=crop_mode,
            deterministic=deterministic,
            seed=seed,
            break_clusters=break_clusters,
            dx_token=self.vocab_size if additional_dx_token else 1,
            whitelist_tokens=np.concatenate(
                (np.array([NO_EVENT_TOKEN]), self.sex_tokens, self.lifestyle_tokens)
            ),
        )

    def __len__(self):
        return self.participants.size

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @cached_property
    def detokenizer(self):
        return {v: k for k, v in self.tokenizer.items()}

    @property
    def lifestyle_tokens(self):
        return np.array([self.tokenizer[i] for i in LIFESTYLE])

    @property
    def sex_tokens(self):
        return np.array([self.tokenizer[i] for i in SEX])

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        x_pid, t_pid = self.reader[pid]

        x_pid, t_pid = self.token_transform(x_pid, t_pid)

        return x_pid[:-1], t_pid[:-1], x_pid[1:], t_pid[1:]

    def subset_participants_for_prompt(
        self, prompt_age: None | float, prompt_tokens: None | np.ndarray
    ):
        keep_lst = list()
        for i in range(self.participants.size):
            x_pid, t_pid, _, _ = self[i]
            tokens = x_pid.copy()
            age = t_pid.copy()
            if prompt_age is not None:
                if age.min() > prompt_age:
                    continue
                tokens = tokens[age <= prompt_age]
            if prompt_tokens is not None:
                if not np.isin(tokens, prompt_tokens).any():
                    continue
            keep_lst.append(i)
        print(f"{len(keep_lst)}/{self.participants.size} participants remaining")
        self.participants = self.participants[keep_lst]

    def get_batch(self, batch_idx: Iterable):

        X0, T0, X1, T1 = list(), list(), list(), list()
        for idx in batch_idx:
            x0, t0, x1, t1 = self[idx]
            X0.append(x0)
            X1.append(x1)
            T0.append(t0)
            T1.append(t1)

        X0 = collate_batch(X0)
        T0 = collate_batch(T0, fill_value=-1e4)
        X1 = collate_batch(X1)
        T1 = collate_batch(T1, fill_value=-1e4)

        X0 = torch.tensor(X0, dtype=torch.long)
        T0 = torch.tensor(T0, dtype=torch.float32)
        X1 = torch.tensor(X1, dtype=torch.long)
        T1 = torch.tensor(T1, dtype=torch.float32)

        return X0, T0, X1, T1


class MultimodalDataset:

    def __init__(
        self,
        reader: Any,
        pids: np.ndarray,
        token_transform: None | Callable = None,
        biomarker_transform: None | Callable = None,
        prompt_transform: None | Callable = None,
    ):

        self.reader = reader
        self.tokenizer = self.reader.tokenizer

        self.participants = pids
        self.token_transform = token_transform
        self.biomarker_transform = biomarker_transform
        self.prompt_transform = prompt_transform

    def __len__(self):
        return self.participants.size

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def detokenizer(self):
        return {v: k for k, v in self.tokenizer.items()}

    @property
    def expansion_tokens(self):
        tokens = list()
        for exp_pack in self.reader.expansion_packs.values():
            tokens.extend([v + exp_pack.offset for v in exp_pack.tokenizer.values()])
        return tokens

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        x, t, bio_x_dict, bio_t, bio_m = self.reader[pid]

        if self.token_transform is not None:
            x, t = self.token_transform(x, t)

        if self.biomarker_transform is not None:
            bio_x_dict, bio_t, bio_m = self.biomarker_transform(
                bio_x_dict, bio_t, bio_m
            )

        if self.prompt_transform is not None:
            x0, t0, bio_x_dict, bio_t, bio_m, x1, t1 = self.prompt_transform(
                x, t, bio_x_dict, bio_t, bio_m
            )
        else:
            x0, x1 = x[:-1].copy(), x[1:].copy()
            t0, t1 = t[:-1].copy(), t[1:].copy()

        return x0, t0, bio_x_dict, bio_t, bio_m, x1, t1

    def get_batch(self, batch_idx: Iterable):

        X0, T0, X1, T1 = list(), list(), list(), list()
        bio_X_dict, bio_T, bio_M = defaultdict(list), list(), list()
        for idx in batch_idx:
            x0, t0, bio_x_dict, bio_t, bio_m, x1, t1 = self[idx]
            X0.append(x0)
            T0.append(t0)
            X1.append(x1)
            T1.append(t1)

            for modality in bio_x_dict.keys():
                bio_X_dict[modality].extend(bio_x_dict[modality])
            bio_T.append(bio_t)
            bio_M.append(bio_m)

        X0 = collate_batch(X0)
        X0 = torch.tensor(X0, dtype=torch.long)
        T0 = collate_batch(T0, fill_value=-1e4)
        T0 = torch.tensor(T0, dtype=torch.float32)
        X1 = collate_batch(X1)
        X1 = torch.tensor(X1, dtype=torch.long)
        T1 = collate_batch(T1, fill_value=-1e4)
        T1 = torch.tensor(T1, dtype=torch.float32)

        for modality, bio_x_lst in bio_X_dict.items():
            bio_X_dict[modality] = torch.from_numpy(np.stack(bio_x_lst))  # type: ignore
        bio_T = collate_batch(bio_T, fill_value=-1e4)
        bio_T = torch.tensor(bio_T, dtype=torch.float32)
        bio_M = collate_batch(bio_M)
        bio_M = torch.tensor(bio_M, dtype=torch.long)

        return X0, T0, bio_X_dict, bio_T, bio_M, X1, T1
