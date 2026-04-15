import functools
import os
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
import yaml

from delphi.data.utils import (
    append_no_event,
    collate_batch,
    crop_contiguous,
    dissolve_clusters,
    dropout_biomarkers,
    exclude_tokens,
    identity_transform,
    perturb_time,
    sort_by_time,
    update_tokenizer,
)
from delphi.env import DELPHI_DATA_DIR
from delphi.multimodal import Modality

LIFESTYLE = [
    "bmi_low",
    "bmi_mid",
    "bmi_high",
    "smoking_low",
    "smoking_mid",
    "smoking_high",
    "alcohol_low",
    "alcohol_mid",
    "alcohol_high",
]

SEX = ["female", "male"]

NO_EVENT_TOKEN = 1


def cut_prompt(
    idx: torch.Tensor,
    age: torch.Tensor,
    prompt_age: None | float | torch.Tensor,
    prompt_token: None | torch.Tensor,
    append_no_event: bool,
):

    idx = idx.clone()
    age = age.clone()

    if prompt_age is None:
        assert prompt_token is not None
        is_prompt = torch.isin(idx, prompt_token)
        assert is_prompt.any(dim=1).all(), "found sequences with no prompt_token(s)"
        prompt_age = age.clone()
        prompt_age[~is_prompt] = -10000
        prompt_age = prompt_age.max(dim=1, keepdim=True)[0]

    if isinstance(prompt_age, float):
        prompt_age = torch.full((idx.shape[0], 1), fill_value=prompt_age)

    idx[age > prompt_age] = 0
    age[age > prompt_age] = -10000.0

    if append_no_event:
        idx = torch.nn.functional.pad(idx, (0, 1), "constant", 1)
        age = torch.cat((age, age.max(dim=1, keepdim=True)[0]), dim=1)

    age_sort = age.argsort(1)
    idx = idx.gather(1, age_sort)
    age = age.gather(1, age_sort)

    trim_margin = torch.min(torch.sum(idx == 0, dim=1)).item()
    idx, age = idx[:, trim_margin:], age[:, trim_margin:]

    return idx, age, prompt_age


def cut_prompt_multimodal(
    idx: torch.Tensor,
    age: torch.Tensor,
    bio_x_dict: dict,
    bio_t: torch.Tensor,
    bio_m: torch.Tensor,
    prompt_age: None | float | torch.Tensor = None,
    prompt_token: None | torch.Tensor = None,
    prompt_modalities: None | list = None,
    append_no_event: bool = False,
):
    """Cut prompt for multimodal data.

    Exactly one of prompt_age, prompt_token, or prompt_modalities must be set.

    prompt_modalities: list of Modality enums. The prompt age is derived from
        the last biomarker measurement time among these modalities, per sample.
    """

    n_set = sum(x is not None for x in [prompt_age, prompt_token, prompt_modalities])
    assert n_set == 1, f"exactly one cutting criterion must be set, got {n_set}"

    if prompt_modalities is not None:
        mod_values = torch.tensor(
            [m.value if hasattr(m, "value") else m for m in prompt_modalities],
            device=bio_m.device,
        )
        is_prompt_mod = torch.isin(bio_m, mod_values)
        assert is_prompt_mod.any(
            dim=1
        ).all(), "found samples with no measurements for prompt_modalities"
        prompt_age = bio_t.clone()
        prompt_age[~is_prompt_mod] = -1e4
        prompt_age = prompt_age.max(dim=1, keepdim=True)[0]

    # cut disease tokens; pass prompt_token only when that's the criterion
    idx, age, prompt_age = cut_prompt(
        idx,
        age,
        prompt_age=prompt_age if prompt_token is None else None,
        prompt_token=prompt_token,
        append_no_event=append_no_event,
    )

    # cut biomarker data at prompt_age (now always a (B, 1) tensor)
    keep = (bio_m > 0) & (bio_t <= prompt_age)

    new_bio_x_dict = {}
    for modality, bio_x in bio_x_dict.items():
        old_mask = (bio_m == modality.value).flatten()
        new_mask = (keep & (bio_m == modality.value)).flatten()
        keep_in_bio_x = new_mask[old_mask]
        bio_x = bio_x[keep_in_bio_x]
        if bio_x.shape[0] > 0:
            new_bio_x_dict[modality] = bio_x

    bio_t = bio_t.clone()
    bio_m = bio_m.clone()
    bio_t[~keep] = -1e4
    bio_m[~keep] = 0

    sort_idx = bio_t.argsort(dim=1)
    bio_t = bio_t.gather(1, sort_idx)
    bio_m = bio_m.gather(1, sort_idx)

    trim = torch.min(torch.sum(bio_m == 0, dim=1)).item()
    if trim > 0:
        bio_t = bio_t[:, trim:]
        bio_m = bio_m[:, trim:]

    return idx, age, new_bio_x_dict, bio_t, bio_m, prompt_age


def load_label_meta(data_dir="ukb_real_data") -> pd.DataFrame:
    """Load disease label metadata (ICD chapters, colors)."""
    path = Path(DELPHI_DATA_DIR) / data_dir / "labels_chapters_colours.csv"
    return pd.read_csv(path)


class UKBDataset:

    def __init__(
        self,
        data_dir: str = "ukb_real_data",
        subject_list: str = "participants/train_fold.bin",
        no_event_interval: None | float = 5 * 365.25,
        no_event_mode: str = "legacy-random",
        block_size: None | int = None,
        perturb: bool = True,
        perturb_list: None | list = None,
        exclude: bool = False,
        exclude_list: None | list = None,
        crop_mode: Literal["left", "right", "random"] = "right",
        break_clusters: bool = False,
        additional_dx_token: bool = True,
        seed: int = 42,
        deterministic: bool = False,
        memmap: bool = False,
    ):

        self._init_args = locals().copy()
        self._init_args.pop("self")  # Remove 'self' reference

        (
            self.tokenizer,
            self.start_pos,
            self.seq_len,
            self.participants,
            self.tokens,
            self.time_steps,
        ) = load_core_data_package(
            data_dir=data_dir, subject_list=subject_list, memmap=memmap
        )

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic

        self.no_event_interval = no_event_interval
        if no_event_interval is not None:
            self.append_no_event = functools.partial(
                append_no_event,
                interval=no_event_interval,
                token=self.tokenizer["no_event"],
                mode=no_event_mode,
            )
        else:
            self.append_no_event = identity_transform

        if exclude:
            if exclude_list is None:
                exclude_list = LIFESTYLE
            tokens_to_exclude = np.array(
                [self.tokenizer[event] for event in exclude_list]
            )
            self.exclude_tokens = functools.partial(
                exclude_tokens, blacklist=tokens_to_exclude
            )
        else:
            self.exclude_tokens = identity_transform

        if perturb:
            if perturb_list is None:
                perturb_list = LIFESTYLE
            tokens_to_perturb = np.array(
                [self.tokenizer[event] for event in perturb_list]
            )
            self.perturb_time = functools.partial(
                perturb_time,
                tokens=tokens_to_perturb,
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
            if additional_dx_token:
                self.dx_token = self.vocab_size
                self.tokenizer["dx"] = len(self.tokenizer)
            else:
                self.dx_token = NO_EVENT_TOKEN
            self.break_clusters = functools.partial(
                dissolve_clusters,
                whitelist=np.concatenate(
                    (np.array([NO_EVENT_TOKEN]), self.sex_tokens, self.lifestyle_tokens)
                ),
                dx_token=self.dx_token,
            )
        else:
            self.break_clusters = identity_transform

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
        if self.deterministic:
            rng = np.random.default_rng(pid + self.seed)
        else:
            rng = self.rng
        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l].astype(np.uint32)
        t_pid = self.time_steps[i : i + l].astype(np.float32)
        x_pid, t_pid = self.exclude_tokens(x_pid, t_pid)
        x_pid, t_pid = self.append_no_event(x_pid, t_pid, rng=rng)
        x_pid, t_pid = self.perturb_time(x_pid, t_pid, rng=rng)
        t_pid, x_pid = sort_by_time(t_pid, x_pid, stable=self.deterministic)
        x_pid, t_pid = self.crop_block_size(x_pid, t_pid, rng=rng)
        x_pid, t_pid = self.break_clusters(x_pid, t_pid, rng=rng)

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


class Biomarker:

    def __init__(
        self,
        path: str,
        stats_subjects: None | np.ndarray = None,
        memmap: bool = False,
        first_time_only: bool = True,
        z_score: bool = False,
    ):

        self.path = path
        data_path = os.path.join(path, "data.bin")
        if memmap:
            self.data = np.memmap(data_path, dtype=np.float32, mode="r")
        else:
            self.data = np.fromfile(data_path, dtype=np.float32)

        with open(os.path.join(path, "features.yaml"), "r") as f:
            self.features = yaml.safe_load(f)
        self.feat2idx = {feat: i for i, feat in enumerate(self.features)}
        self.n_features = len(self.features)

        p2i = pd.read_csv(os.path.join(path, "p2i.csv")).sort_values(by=["pid", "time"])
        self.start_pos = p2i["start_pos"].to_numpy()
        self.seq_len = p2i["seq_len"].to_numpy()
        self.time_steps = p2i["time"].to_numpy()
        self.pids = p2i["pid"].to_numpy()
        self.uniq_pids, ct = np.unique(p2i["pid"].to_numpy(), return_counts=True)
        cumul_ct = np.insert(np.cumsum(ct), 0, 0, axis=0)
        self.pid2idx = dict(zip(self.uniq_pids, cumul_ct))
        self.pid2cnt = dict(zip(self.uniq_pids, ct))

        self.first_time_only = first_time_only
        self.z_score = z_score
        if stats_subjects is None:
            stats_subjects = self.uniq_pids
        self.mean, self.std = self.stats(stats_subjects)

    def __repr__(self):
        return f"Biomarker(path={self.path}, n_features={self.n_features})"

    def to_array(self, subjects):
        data, subs = list(), list()
        include = np.isin(self.pids, subjects)
        start_pos = self.start_pos[include]
        seq_len = self.seq_len[include]
        pids = self.pids[include]
        seen = set()
        for pid, i, l in zip(pids, start_pos, seq_len):
            if self.first_time_only:
                if pid in seen:
                    continue
                seen.add(pid)
            pid_data = self.data[i : i + l]
            data.append(pid_data)
            subs.append(pid)
        data = np.stack(data, axis=0)
        return data, np.array(subs)

    @property
    def background(self):
        if self.z_score:
            return np.zeros((self.n_features,))
        else:
            return self.mean

    def stats(self, subjects: np.ndarray):
        data, _ = self.to_array(subjects)
        return np.mean(data, axis=0), np.std(data, axis=0)

    def transform(self, x):
        if self.z_score:
            return (x - self.mean) / self.std
        return x

    def untransform(self, z):
        if self.z_score:
            return z * self.std + self.mean
        return z

    def first_occurrence_times(self, pids: np.ndarray) -> np.ndarray:
        """Return the timestamp of the first measurement for each pid.

        Returns NaN for pids that have no measurement in this biomarker.
        """
        result = np.full(len(pids), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in self.pid2idx:
                result[i] = self.time_steps[self.pid2idx[pid]]
        return result

    def __getitem__(
        self, pid: int
    ) -> tuple[None | list[np.ndarray], None | np.ndarray]:

        if pid not in self.pid2idx:
            return None, None

        pid_i = self.pid2idx[pid]
        pid_l = self.pid2cnt[pid]
        pid_slice = slice(pid_i, pid_i + pid_l)

        pid_data = list()
        pid_time = self.time_steps[pid_slice]
        pid_seq_len = self.seq_len[pid_slice]
        pid_start_pos = self.start_pos[pid_slice]
        for i, l in zip(pid_start_pos, pid_seq_len):
            x = self.data[i : i + l]
            x = self.transform(x)
            pid_data.append(x)
            if self.first_time_only:
                pid_time = pid_time[[0]]
                break
        return pid_data, pid_time


class ExpansionPack:

    def __init__(self, path: str, offset: int, memmap: bool = False):

        p2i = pd.read_csv(os.path.join(path, "p2i.csv"), index_col="pid")
        self.pids = p2i.index.to_numpy()
        self.offset = offset
        self.start_pos = p2i["start_pos"].to_dict()
        self.seq_len = p2i["seq_len"].to_dict()
        data_path = os.path.join(path, "data.bin")
        time_path = os.path.join(path, "time.bin")
        if memmap:
            self.tokens = np.memmap(data_path, dtype=np.uint32, mode="r")
            self.time_steps = np.memmap(time_path, dtype=np.uint32, mode="r")
        else:
            self.tokens = np.fromfile(data_path, dtype=np.uint32)
            self.time_steps = np.fromfile(data_path, dtype=np.uint32)

        tokenizer_path = os.path.join(path, "tokenizer.yaml")
        with open(tokenizer_path, "r") as f:
            self.tokenizer = yaml.safe_load(f)

    def __getitem__(self, pid: int) -> tuple[np.ndarray, np.ndarray]:

        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l] + self.offset
        t_pid = self.time_steps[i : i + l]

        return x_pid, t_pid


class MultimodalUKBDataset:

    def __init__(
        self,
        data_dir: str = "ukb_real_data",
        expansion_pack_dir: str = "expansion_packs",
        expansion_packs: None | list = None,
        biomarker_datasets: None | dict = None,
        biomarker_dir: str = "biomarkers",
        biomarkers: None | list = None,
        z_score_biomarkers: bool = True,
        first_time_only: bool = True,
        must_have_biomarkers: None | list = None,
        biomarker_require: str = "all",
        stats_subject_list: None | str = None,
        subject_list: str = "participants/train_fold.bin",
        must_have_expansion_packs: None | list = None,
        expansion_pack_require: str = "all",
        no_event_interval: None | float = 5 * 365.25,
        no_event_mode: str = "legacy-random",
        perturb: bool = False,
        perturb_list: None | list = None,
        block_size: None | int = None,
        crop_mode: Literal["left", "right", "random"] = "left",
        biomarker_dropout: None | float = None,
        seed: int = 42,
        deterministic: bool = False,
        memmap: bool = False,
    ):
        """
        args:
            data_dir: directory name of UKB dataset
            expansion_pack_dir: sub-directory within data_dir containing data for expansion packs
            expansion_packs: a list of expansion packs to include
            biomarker_datasets: a list of pre-initialized Biomarkers
            biomarker_dir: sub-directory within data_dir containing biomarker data
            biomarkers: a list of biomarkers to load
            z_score_biomarkers: whether to z-score biomarker values
            first_time_only: if True only use the first occurrence of each biomarker
            must_have_biomarkers: a list of biomarkers that each participant must have to be included
            biomarker_require: "all" to keep participants with all listed biomarkers, "any" to keep those with at least one
            stats_subject_list: sub-path within data_dir to an array of subjects for computing Biomarker stats
            subject_list: sub-path within data_dir to an array of subjects for loading
            no_event_interval: average time intervals for introducing no-event tokens
            no_event_mode: mode for introducing no-event tokens
                refer to append_no_event for more details
            perturb: whether to perturb timestamps of tokens in perturb_list
            perturb_list: a list of tokens whose timestamps are perturbed for data augmentation
                default is the lifestyle tokens in the UKB
            block_size: maximum sequence length
            crop_mode: where to start cropping a sequence that exceeds block_size
                "left": start from the beginning
                "right": start from the end
                "random": start from a random position in the middle
            biomarker_dropout: if not None, each biomarker measurement is independently dropped
                with this probability during __getitem__
            seed: random seed for reproducibility
            deterministic: if True, the same participant will always receive the same augmentations.
            memmap: whether to load data files in memmap mode
        note: the following defaults differ from UKBDataset
            - crop_mode
            - perturb
        """

        self._init_args = locals().copy()
        self._init_args.pop("self")  # Remove 'self' reference

        (
            self.tokenizer,
            self.start_pos,
            self.seq_len,
            self.participants,
            self.tokens,
            self.time_steps,
        ) = load_core_data_package(
            data_dir=data_dir, subject_list=subject_list, memmap=memmap
        )

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic

        self.expansion_pack_dir = os.path.join(
            DELPHI_DATA_DIR, data_dir, expansion_pack_dir
        )
        self.expansion_packs = dict()
        self.expansion_pack_tokenizers = dict()
        if expansion_packs is not None:
            expansion_packs.sort()
            for pack in expansion_packs:
                pack_path = os.path.join(self.expansion_pack_dir, pack)
                assert os.path.exists(pack_path), FileNotFoundError(
                    f"expansion pack {pack_path} not found"
                )
                tokenizer_path = os.path.join(pack_path, "tokenizer.yaml")
                with open(tokenizer_path, "r") as f:
                    add_tokenizer = yaml.safe_load(f)

                self.tokenizer, offset = update_tokenizer(
                    base_tokenizer=self.tokenizer, add_tokenizer=add_tokenizer  # type: ignore
                )
                self.expansion_pack_tokenizers[pack] = add_tokenizer
                self.expansion_packs[pack] = ExpansionPack(
                    path=pack_path, offset=offset, memmap=memmap
                )

        self.biomarker_dir = os.path.join(DELPHI_DATA_DIR, data_dir, biomarker_dir)
        if biomarker_datasets is not None:
            assert biomarkers is None
            self.mod_ds = biomarker_datasets
        else:
            if stats_subject_list is None:
                stats_subjects = self.participants
            else:
                stats_subjects = np.fromfile(
                    os.path.join(DELPHI_DATA_DIR, data_dir, stats_subject_list),
                    dtype=np.uint32,
                )

            self.mod_ds = {}
            if biomarkers is not None:
                for modality in biomarkers:
                    modality = Modality[modality.upper()]
                    biomarker_path = os.path.join(
                        self.biomarker_dir, modality.name.lower()
                    )
                    dataset = Biomarker(
                        path=biomarker_path,
                        stats_subjects=stats_subjects,
                        memmap=memmap,
                        first_time_only=first_time_only,
                        z_score=z_score_biomarkers,
                    )
                    self.mod_ds[modality] = dataset

        if must_have_biomarkers is not None:
            print(
                f"keeping participants with {biomarker_require} of: {must_have_biomarkers}"
            )
            old_n = self.participants.size
            if biomarker_require == "all":
                for mod in must_have_biomarkers:
                    mod_pids = self.get_modality_participants(mod)
                    self.participants = self.participants[
                        np.isin(self.participants, mod_pids)
                    ]
            elif biomarker_require == "any":
                union_pids = np.concatenate(
                    [
                        self.get_modality_participants(mod)
                        for mod in must_have_biomarkers
                    ]
                )
                self.participants = self.participants[
                    np.isin(self.participants, union_pids)
                ]
            else:
                raise ValueError(
                    f"biomarker_require must be 'all' or 'any', got '{biomarker_require}'"
                )
            print(f"{self.participants.size}/{old_n} remaining")

        if must_have_expansion_packs is not None:
            old_n = self.participants.size
            if expansion_pack_require == "all":
                for pack in must_have_expansion_packs:
                    exp_pids = self.get_expansion_pack_participants(pack)
                    self.participants = self.participants[
                        np.isin(self.participants, exp_pids)
                    ]
            elif expansion_pack_require == "any":
                union_pids = np.concatenate(
                    [
                        self.get_expansion_pack_participants(pack)
                        for pack in must_have_expansion_packs
                    ]
                )
                self.participants = self.participants[
                    np.isin(self.participants, union_pids)
                ]
            else:
                raise ValueError(
                    f"expansion_pack_require must be 'all' or 'any', got '{biomarker_require}'"
                )
            print(f"{self.participants.size}/{old_n} remaining")

        if no_event_interval is not None:
            self.append_no_event = functools.partial(
                append_no_event,
                interval=no_event_interval,
                token=self.tokenizer["no_event"],
                mode=no_event_mode,
            )
        else:
            self.append_no_event = identity_transform

        if perturb:
            if perturb_list is None:
                perturb_list = LIFESTYLE
            tokens_to_perturb = np.array(
                [self.tokenizer[event] for event in perturb_list]
            )
            self.perturb_time = functools.partial(
                perturb_time, tokens=tokens_to_perturb
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

        if biomarker_dropout is not None:
            self.dropout_biomarkers = functools.partial(
                dropout_biomarkers,
                p=biomarker_dropout,
            )
        else:
            self.dropout_biomarkers = identity_transform

    def get_modality_participants(self, modality) -> np.ndarray:
        if isinstance(modality, str):
            modality = Modality[modality.upper()]
        return pd.read_csv(
            os.path.join(self.biomarker_dir, modality.name.lower(), "p2i.csv")
        )["pid"].to_numpy()

    def get_expansion_pack_participants(self, expansion_pack) -> np.ndarray:
        return pd.read_csv(
            os.path.join(self.expansion_pack_dir, expansion_pack.lower(), "p2i.csv")
        )["pid"].to_numpy()

    def first_occurrence_times(self, modality):
        if isinstance(modality, str):
            modality = Modality[modality.upper()]
        p2i = pd.read_csv(
            os.path.join(self.biomarker_dir, modality.name.lower(), "p2i.csv")
        ).sort_values(by=["pid", "time"])
        first = p2i.groupby("pid")["time"].first()
        result = np.full(len(self.participants), np.nan, dtype=np.float32)
        mask = np.isin(self.participants, first.index)
        result[mask] = first.reindex(self.participants[mask]).to_numpy()
        return result

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
        for exp_pack in self.expansion_packs.values():
            tokens.extend([v + exp_pack.offset for v in exp_pack.tokenizer.values()])
        return tokens

    def __getitem__(self, idx: int):

        pid = self.participants[idx]
        if self.deterministic:
            rng = np.random.default_rng(pid + self.seed)
        else:
            rng = self.rng
        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x = self.tokens[i : i + l].astype(np.uint32)
        t = self.time_steps[i : i + l].astype(np.float32)
        x_lst, t_lst = [x], [t]
        for expansion_pack in self.expansion_packs.values():
            exp_x, exp_t = expansion_pack[pid]
            x_lst.append(exp_x)
            t_lst.append(exp_t)
        x = np.concatenate(x_lst)
        t = np.concatenate(t_lst)
        x, t = self.append_no_event(x, t, rng=rng)
        x, t = self.perturb_time(x, t, rng=rng)
        t, x = sort_by_time(t, x)
        x, t = self.crop_block_size(x, t, rng=rng)

        bio_x_dict = dict()
        bio_t_lst = list()
        bio_m_lst = list()
        for modality, ds in self.mod_ds.items():
            bio_x, mod_t = ds[pid]
            if bio_x is None:
                continue
            bio_x_dict[modality] = bio_x
            mod_m = np.full_like(mod_t, fill_value=modality.value)
            bio_t_lst.append(mod_t)
            bio_m_lst.append(mod_m)

        if len(bio_x_dict) == 0:
            assert len(bio_t_lst) == 0
            assert len(bio_m_lst) == 0
            bio_t = np.array([])
            bio_m = np.array([])
        else:
            bio_t = np.concatenate(bio_t_lst)
            bio_m = np.concatenate(bio_m_lst)

            bio_t, bio_m = sort_by_time(bio_t, bio_m)

        bio_x_dict, bio_t, bio_m = self.dropout_biomarkers(
            bio_x_dict, bio_t, bio_m, rng=rng
        )

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


def load_core_data_package(data_dir: str, subject_list, memmap: bool = False):

    dataset_dir = Path(DELPHI_DATA_DIR) / data_dir
    tokenizer_path = dataset_dir / "tokenizer.yaml"
    with open(tokenizer_path, "r") as f:
        tokenizer = yaml.safe_load(f)

    p2i = pd.read_csv(dataset_dir / "p2i.csv", index_col="pid")
    start_pos = p2i["start_pos"].to_dict()
    seq_len = p2i["seq_len"].to_dict()

    if isinstance(subject_list, (str, os.PathLike)):
        participants_path = dataset_dir / subject_list
        if memmap:
            participants = np.memmap(participants_path, dtype=np.uint32, mode="r")
        else:
            participants = np.fromfile(participants_path, dtype=np.uint32)
    else:
        participants = np.asarray(subject_list, dtype=np.uint32)

    tokens_path = dataset_dir / "data.bin"
    time_steps_path = dataset_dir / "time.bin"
    if memmap:
        tokens = np.memmap(tokens_path, dtype=np.uint32, mode="r")
        timesteps = np.memmap(time_steps_path, dtype=np.uint32, mode="r")
    else:
        tokens = np.fromfile(tokens_path, dtype=np.uint32)
        timesteps = np.fromfile(time_steps_path, dtype=np.uint32)

    return tokenizer, start_pos, seq_len, participants, tokens, timesteps
