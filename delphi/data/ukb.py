import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from delphi.data.reader import MultimodalReader, TokenReader
from delphi.env import DELPHI_DATA_READ as DELPHI_DATA_DIR

NO_EVENT_TOKEN = 1


class UKBReader(TokenReader):
    base_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data"
    bmi_keys = [
        "bmi_low",
        "bmi_mid",
        "bmi_high",
    ]
    smoking_keys = [
        "smoking_low",
        "smoking_mid",
        "smoking_high",
    ]
    alcohol_keys = [
        "alcohol_low",
        "alcohol_mid",
        "alcohol_high",
    ]
    lifestyle_keys = bmi_keys + smoking_keys + alcohol_keys
    sex_keys = ["female", "male"]

    def __init__(self, memmap: bool = False):

        tokenizer_path = self.base_dir / "tokenizer.yaml"
        with open(tokenizer_path, "r") as f:
            tokenizer = yaml.safe_load(f)

        self.p2i = pd.read_csv(self.base_dir / "p2i.csv", index_col="pid")
        start_pos = self.p2i["start_pos"].to_dict()
        seq_len = self.p2i["seq_len"].to_dict()

        tokens_path = self.base_dir / "data.bin"
        time_steps_path = self.base_dir / "time.bin"
        if memmap:
            tokens = np.memmap(tokens_path, dtype=np.uint32, mode="r")
            timesteps = np.memmap(time_steps_path, dtype=np.uint32, mode="r")
        else:
            tokens = np.fromfile(tokens_path, dtype=np.uint32)
            timesteps = np.fromfile(time_steps_path, dtype=np.uint32)

        super().__init__(tokens, timesteps, start_pos, seq_len, tokenizer)

    @classmethod
    def participants(cls, fold):
        return np.fromfile(
            cls.base_dir / "participants" / f"{fold}_fold.bin", dtype=np.uint32
        )

    @classmethod
    def labels(cls) -> pd.DataFrame:
        """Load disease label metadata (ICD chapters, colors)."""
        return pd.read_csv(cls.base_dir / "labels_chapters_colours.csv")

    def is_female(self, pids: np.ndarray) -> np.ndarray:
        female_token = self.tokenizer["female"]
        out = np.zeros(len(pids), dtype=bool)
        for i, pid in enumerate(pids):
            start = self.start_pos[int(pid)]
            length = self.seq_len[int(pid)]
            out[i] = (self.tokens[start : start + length] == female_token).any()
        return out

    def recruitment_times(self, pids: np.ndarray) -> np.ndarray:
        event_times = self.event_times(pids)
        lifestyle_tokens = np.array([self.tokenizer[e] for e in self.lifestyle_keys])
        return np.nanmin(event_times[:, lifestyle_tokens], axis=1)


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


class ExpansionPack(TokenReader):

    base_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "expansion_packs"

    def __init__(self, name: str, memmap: bool = False):

        path = self.base_dir / name
        assert os.path.exists(path), FileNotFoundError(
            f"expansion pack {path} not found"
        )
        p2i = pd.read_csv(os.path.join(path, "p2i.csv"), index_col="pid")
        self.pids = p2i.index.to_numpy()
        start_pos = p2i["start_pos"].to_dict()
        seq_len = p2i["seq_len"].to_dict()
        data_path = os.path.join(path, "data.bin")
        time_path = os.path.join(path, "time.bin")
        if memmap:
            tokens = np.memmap(data_path, dtype=np.uint32, mode="r")
            timesteps = np.memmap(time_path, dtype=np.uint32, mode="r")
        else:
            tokens = np.fromfile(data_path, dtype=np.uint32)
            timesteps = np.fromfile(time_path, dtype=np.uint32)

        tokenizer_path = os.path.join(path, "tokenizer.yaml")
        with open(tokenizer_path, "r") as f:
            tokenizer = yaml.safe_load(f)

        super().__init__(tokens, timesteps, start_pos, seq_len, tokenizer)

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
                result[i] = pack.timesteps[pack.start_pos[pid]]
        return result


class MultimodalUKBReader(MultimodalReader):
    token_reader: UKBReader

    bmi_keys = UKBReader.bmi_keys
    smoking_keys = UKBReader.smoking_keys
    alcohol_keys = UKBReader.alcohol_keys
    lifestyle_keys = UKBReader.lifestyle_keys
    sex_keys = UKBReader.sex_keys

    def __init__(
        self,
        expansion_packs: list[str] | None = None,
        biomarkers: list[str] | dict[str, int] | None = None,
        memmap: bool = False,
    ):
        """
        args:
            expansion_packs: a list of expansion packs to include
            biomarkers: either a list of biomarker names (sorted and assigned
                indices starting at RESERVED_MOD_IDX), or a {name: idx} mapping
                to use as-is (e.g. loaded from a checkpoint). Keys/names are
                normalized to lowercase.
            memmap: whether to load data files in memmap mode
        """
        bm_names, biomarker2idx = self._normalize_biomarkers(biomarkers)
        super().__init__(
            token_reader=UKBReader(memmap=memmap),
            expansion_packs={
                n: ExpansionPack(name=n, memmap=memmap) for n in expansion_packs or []
            },
            biomarkers={n: Biomarker(name=n, memmap=memmap) for n in bm_names},
            biomarker2idx=biomarker2idx,
        )

    @classmethod
    def participants(cls, fold):
        return UKBReader.participants(fold)

    @classmethod
    def labels(cls):
        return UKBReader.labels()

    def is_female(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.is_female(pids)

    def event_times(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.event_times(pids)

    def exit_times(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.exit_times(pids)

    def recruitment_times(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.recruitment_times(pids)


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


def filter_participants_with_modalities(pids, biomarkers, expansion_packs):
    if biomarkers is not None:
        total = pids.size
        pids = filter_participants_with_biomarkers(
            pids, biomarkers=biomarkers, any=True
        )
        print(f"{pids.size} / {total} val pids (biomarker filter)")
    if expansion_packs is not None:
        total = pids.size
        pids = filter_participants_with_expansion_packs(
            pids, expansion_packs=expansion_packs, any=True
        )
        print(f"{pids.size} / {total} val pids (expansion pack filter)")

    return pids


def first_modality_timestep(pids, biomarkers, expansion_packs):

    cutoff = np.full(len(pids), np.nan, dtype=np.float32)
    for mod_name in biomarkers or []:
        first = Biomarker.first_occurrence_times(mod_name, pids)
        cutoff = np.fmin(cutoff, first)
    for pack_name in expansion_packs or []:
        first = ExpansionPack.first_occurrence_times(pack_name, pids)
        cutoff = np.fmin(cutoff, first)

    return cutoff
