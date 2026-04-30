import os
import pprint
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from delphi.data.utils import (
    sort_by_time,
    update_tokenizer,
)
from delphi.env import DELPHI_DATA_READ as DELPHI_DATA_DIR
from delphi.multimodal import Modality

NO_EVENT_TOKEN = 1


class UKBReader:
    base_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data"
    lifestyle_keys = [
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
    sex_keys = ["female", "male"]

    def __init__(self, memmap: bool = False):

        tokenizer_path = self.base_dir / "tokenizer.yaml"
        with open(tokenizer_path, "r") as f:
            self.tokenizer = yaml.safe_load(f)
        self.vocab_size = len(self.tokenizer)

        self.p2i = pd.read_csv(self.base_dir / "p2i.csv", index_col="pid")
        self.start_pos = self.p2i["start_pos"].to_dict()
        self.seq_len = self.p2i["seq_len"].to_dict()

        tokens_path = self.base_dir / "data.bin"
        time_steps_path = self.base_dir / "time.bin"
        if memmap:
            self.tokens = np.memmap(tokens_path, dtype=np.uint32, mode="r")
            self.timesteps = np.memmap(time_steps_path, dtype=np.uint32, mode="r")
        else:
            self.tokens = np.fromfile(tokens_path, dtype=np.uint32)
            self.timesteps = np.fromfile(time_steps_path, dtype=np.uint32)

    @classmethod
    def participants(cls, fold):
        return np.fromfile(
            cls.base_dir / "participants" / f"{fold}_fold.bin", dtype=np.uint32
        )

    @classmethod
    def labels(cls) -> pd.DataFrame:
        """Load disease label metadata (ICD chapters, colors)."""
        return pd.read_csv(cls.base_dir / "labels_chapters_colours.csv")

    @cached_property
    def detokenizer(self):
        return {v: k for k, v in self.tokenizer.items()}

    def __getitem__(self, pid: int):

        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l].astype(np.uint32)
        t_pid = self.timesteps[i : i + l].astype(np.float32)

        return x_pid, t_pid

    def is_female(self, pids: np.ndarray) -> np.ndarray:
        female_token = self.tokenizer["female"]
        out = np.zeros(len(pids), dtype=bool)
        for i, pid in enumerate(pids):
            start = self.start_pos[int(pid)]
            length = self.seq_len[int(pid)]
            out[i] = (self.tokens[start : start + length] == female_token).any()
        return out

    def event_times(self, pids: np.ndarray) -> np.ndarray:
        """N by V array of first-occurrence times; NaN where a token never occurs."""
        out = np.full((len(pids), self.vocab_size), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            start = self.start_pos[int(pid)]
            length = self.seq_len[int(pid)]
            x = self.tokens[start : start + length]
            t = self.timesteps[start : start + length].astype(np.float32)
            uniq, first_idx = np.unique(x, return_index=True)
            out[i, uniq] = t[first_idx]
        return out

    def participants_with_event(self, pids: np.ndarray, event: str) -> np.ndarray:
        token = self.tokenizer[event]
        pids_with_event = list()
        for i, pid in enumerate(pids):
            start = self.start_pos[int(pid)]
            length = self.seq_len[int(pid)]
            x = self.tokens[start : start + length]
            if token in x:
                pids_with_event.append(pid)
        return np.array(pids_with_event)

    def exit_times(self, pids: np.ndarray) -> np.ndarray:
        """N array of last token times (exit / censoring time)."""
        out = np.empty(len(pids), dtype=np.float32)
        for i, pid in enumerate(pids):
            start = self.start_pos[int(pid)]
            length = self.seq_len[int(pid)]
            out[i] = self.timesteps[start + length - 1]
        return out


class MultimodalUKBReader:
    lifestyle_keys = UKBReader.lifestyle_keys
    sex_keys = UKBReader.sex_keys

    def __init__(
        self,
        expansion_packs: list | None = None,
        biomarkers: list | None = None,
        memmap: bool = False,
    ):
        """
        args:
            expansion_packs: a list of expansion packs to include
            biomarkers: a list of biomarkers to load
            memmap: whether to load data files in memmap mode
        """

        self.token_reader = UKBReader(memmap=memmap)
        self.tokenizer = self.token_reader.tokenizer
        self.base_tokenizer = self.tokenizer.copy()

        self.expansion_packs = dict()
        self.expansion_offset = dict()
        if expansion_packs is not None:
            expansion_packs.sort()
            for name in expansion_packs:
                self.expansion_packs[name] = ExpansionPack(name=name, memmap=memmap)
                self.tokenizer, offset = update_tokenizer(
                    base_tokenizer=self.tokenizer,
                    add_tokenizer=self.expansion_packs[name].tokenizer,
                )
                self.expansion_offset[name] = offset
        self.vocab_size = len(self.tokenizer)

        self.biomarkers = dict()
        if biomarkers is not None:
            for biomarker in biomarkers:
                self.biomarkers[Modality[biomarker.upper()]] = Biomarker(
                    name=biomarker.lower(),
                    memmap=memmap,
                )

    @classmethod
    def participants(cls, fold):
        return UKBReader.participants(fold)

    @classmethod
    def labels(cls):
        return UKBReader.labels()

    def describe(self) -> None:
        print(f"{type(self).__name__}:")
        config = {
            "expansion_packs": sorted(self.expansion_packs.keys()),
            "biomarkers": sorted(m.name.lower() for m in self.biomarkers),
        }
        pprint.pp(config)

    @cached_property
    def detokenizer(self):
        return {v: k for k, v in self.tokenizer.items()}

    @property
    def expansion_tokens(self):
        return list(set(self.tokenizer.values()) - set(self.base_tokenizer.values()))

    def is_female(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.is_female(pids)

    def event_times(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.event_times(pids)

    def exit_times(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.exit_times(pids)

    def __getitem__(self, pid: int):

        x, t = self.token_reader[pid]
        x_lst, t_lst = [x], [t]
        for name, expansion_pack in self.expansion_packs.items():
            exp_x, exp_t = expansion_pack[pid]
            x_lst.append(exp_x + self.expansion_offset[name])
            t_lst.append(exp_t)
        x = np.concatenate(x_lst)
        t = np.concatenate(t_lst)

        bio_x_dict = dict()
        bio_t_lst = list()
        bio_m_lst = list()
        for modality, ds in self.biomarkers.items():
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

        return x, t, bio_x_dict, bio_t, bio_m


class Biomarker:

    base_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers"

    def __init__(
        self,
        name: str,
        memmap: bool = False,
        first_time_only: bool = True,
    ):

        path = self.base_dir / name
        assert os.path.exists(path), FileNotFoundError(f"biomarker {path} not found")
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

    @classmethod
    def input_size(cls, name: str):
        with open(cls.base_dir / name / "features.yaml", "r") as f:
            features = yaml.safe_load(f)
        return len(features)

    @classmethod
    def participants(cls, name: str) -> np.ndarray:
        p2i = pd.read_csv(cls.base_dir / name / "p2i.csv")
        return p2i["pid"].unique()  # type: ignore

    @classmethod
    def first_occurrence_times(cls, name: str, pids: np.ndarray) -> np.ndarray:
        """Like first_occurrence_times, but reads p2i.csv without a full instance."""
        p2i = pd.read_csv(cls.base_dir / name / "p2i.csv").sort_values(
            by=["pid", "time"]
        )
        first = p2i.groupby("pid")["time"].first()
        result = np.full(len(pids), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in first.index:
                result[i] = first.loc[pid]
        return result

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

    def stats(self, subjects: np.ndarray):
        data, _ = self.to_array(subjects)
        return np.mean(data, axis=0), np.std(data, axis=0)

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
            pid_data.append(x)
            if self.first_time_only:
                pid_time = pid_time[[0]]
                break
        return pid_data, pid_time


class ExpansionPack:

    base_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "expansion_packs"

    def __init__(self, name: str, memmap: bool = False):

        path = self.base_dir / name
        assert os.path.exists(path), FileNotFoundError(
            f"expansion pack {path} not found"
        )
        p2i = pd.read_csv(os.path.join(path, "p2i.csv"), index_col="pid")
        self.pids = p2i.index.to_numpy()
        self.start_pos = p2i["start_pos"].to_dict()
        self.seq_len = p2i["seq_len"].to_dict()
        data_path = os.path.join(path, "data.bin")
        time_path = os.path.join(path, "time.bin")
        if memmap:
            self.tokens = np.memmap(data_path, dtype=np.uint32, mode="r")
            self.time_steps = np.memmap(time_path, dtype=np.uint32, mode="r")
        else:
            self.tokens = np.fromfile(data_path, dtype=np.uint32)
            self.time_steps = np.fromfile(time_path, dtype=np.uint32)

        tokenizer_path = os.path.join(path, "tokenizer.yaml")
        with open(tokenizer_path, "r") as f:
            self.tokenizer = yaml.safe_load(f)

    @classmethod
    def participants(cls, name: str) -> np.ndarray:
        p2i = pd.read_csv(cls.base_dir / name / "p2i.csv")
        return p2i["pid"].unique()

    @classmethod
    def first_occurrence_times(cls, name: str, pids: np.ndarray) -> np.ndarray:
        pack = cls(name=name, memmap=True)
        result = np.full(len(pids), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in pack.start_pos:
                result[i] = pack.time_steps[pack.start_pos[pid]]
        return result

    def __getitem__(self, pid: int) -> tuple[np.ndarray, np.ndarray]:

        if pid not in self.start_pos:
            return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32)

        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l]
        t_pid = self.time_steps[i : i + l]

        return x_pid, t_pid


def filter_participants(pids, filter_list, any=True):

    if any:
        union = np.concatenate([f for f in filter_list])
        pids = pids[np.isin(pids, union)]
    else:
        for f in filter_list:
            pids = pids[np.isin(pids, f)]
    return pids


def filter_participants_with_biomarkers(pids, biomarkers, any=True):

    filter_list = list()
    for biomarker in biomarkers:
        filter_list.append(Biomarker.participants(biomarker))

    return filter_participants(pids, filter_list, any)


def filter_participants_with_expansion_packs(pids, expansion_packs, any=True):

    filter_list = list()
    for pack in expansion_packs:
        filter_list.append(ExpansionPack.participants(pack))

    return filter_participants(pids, filter_list, any)
