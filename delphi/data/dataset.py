from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np
import torch
from tqdm import tqdm

from delphi.data.utils import collate_batch


class Dataset:

    def __init__(
        self,
        reader: Any,
        pids: np.ndarray,
        token_transform: None | Callable = None,
        prompt_transform: None | Callable = None,
    ):

        self.reader = reader
        self.tokenizer = self.reader.tokenizer

        self.participants = pids
        self.token_transform = token_transform
        self.prompt_transform = prompt_transform

    def __len__(self):
        return self.participants.size

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        x, t = self.reader[pid]

        if self.token_transform is not None:
            x, t = self.token_transform(x, t)

        if self.prompt_transform is not None:
            x0, t0, x1, t1 = self.prompt_transform(x, t, pid=pid)
        else:
            x0, x1 = x[:-1].copy(), x[1:].copy()
            t0, t1 = t[:-1].copy(), t[1:].copy()

        return x0, t0, x1, t1

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

    def sort_by_length(
        self, descending: bool = False, progress: bool = True
    ) -> np.ndarray:
        """Reorder ``self.participants`` by post-transform sequence length and
        return the reordered pid array (which is ``self.participants``).

        Grouping similar-length participants into a batch minimizes padding, so
        the forward pass wastes less compute on padded positions. Length is
        measured through ``__getitem__`` so it reflects the token/biomarker
        transforms (which can change length); this is a full pass over the
        reader (one load per participant, no model forward).

        Returning the new order lets callers keep their own per-participant
        arrays (e.g. ``is_female``, pid arrays) aligned to the dataset's row
        order — they must rebind to the return value.
        """
        lengths = np.empty(len(self), dtype=np.int64)
        iterator = range(len(self))
        if progress:
            iterator = tqdm(iterator, desc="sort_by_length", leave=False)
        for idx in iterator:
            x0, _, _, bio_t, _, _, _ = self[idx]
            lengths[idx] = len(x0) + len(bio_t)
        order = np.argsort(lengths, kind="stable")
        if descending:
            order = order[::-1]
        self.participants = self.participants[order]
        return self.participants

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
                x, t, bio_x_dict, bio_t, bio_m, pid=pid
            )
        else:
            x0, x1 = x[:-1].copy(), x[1:].copy()
            t0, t1 = t[:-1].copy(), t[1:].copy()

        return x0, t0, bio_x_dict, bio_t, bio_m, x1, t1

    def get_batch(self, batch_idx: Iterable):
        return self.collate([self[idx] for idx in batch_idx])

    def collate(self, samples: Iterable):
        """Collate ``__getitem__`` outputs into a padded batch.

        Split out of ``get_batch`` so it can double as a torch ``DataLoader``
        ``collate_fn``: workers run ``__getitem__`` (the CPU-bound prompt build)
        and this pads/stacks their outputs into the batch tensors.
        """
        X0, T0, X1, T1 = list(), list(), list(), list()
        bio_X_dict, bio_T, bio_M = defaultdict(list), list(), list()
        for x0, t0, bio_x_dict, bio_t, bio_m, x1, t1 in samples:
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
