import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from cloudpathlib import AnyPath

from delphi.data.reader import BiomarkerReader, MultimodalReader, TokenReader
from delphi.data.utils import filter_participants, list_subdirs
from delphi.env import DELPHI_DATA_READ as DELPHI_DATA_DIR

NO_EVENT_TOKEN = 1


class UKBReader(TokenReader):
    base_dir = AnyPath(DELPHI_DATA_DIR) / "ukb_real_data"
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
        if fold == "all":
            return pd.read_csv(cls.base_dir / "p2i.csv", usecols=["pid"])[
                "pid"
            ].to_numpy(dtype=np.uint32)
        return np.fromfile(
            cls.base_dir / "participants" / f"{fold}_fold.bin", dtype=np.uint32
        )

    @classmethod
    def labels(cls) -> pd.DataFrame:
        """Load disease label metadata (ICD chapters, colors)."""
        return pd.read_csv(str(cls.base_dir / "labels_chapters_colours.csv"))

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


class Biomarker(BiomarkerReader):
    """UKB biomarker store. Flat ``data.bin`` (float32) + ``p2i.csv`` (pid, time,
    start_pos, seq_len); ``_load`` gathers the ragged-flat rows into the 2-D
    canonical layout. All access logic lives on :class:`BiomarkerReader`."""

    base_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers"
    _marker = "data.bin"

    def _load(self, name, memmap=False):
        path = self.base_dir / name
        assert os.path.exists(path), FileNotFoundError(f"biomarker {path} not found")
        data_path = os.path.join(path, "data.bin")
        if memmap:
            flat = np.memmap(data_path, dtype=np.float32, mode="r")
        else:
            flat = np.fromfile(data_path, dtype=np.float32)

        features = self._read_features(name)
        p2i = pd.read_csv(os.path.join(path, "p2i.csv")).sort_values(by=["pid", "time"])
        start_pos = p2i["start_pos"].to_numpy()
        seq_len = p2i["seq_len"].to_numpy()
        if len(start_pos):
            # each visit is one fixed-width (== n_features) feature vector; gather
            # the flat slices, in (pid, time) order, into a 2-D measurement matrix
            values = np.stack([flat[i : i + l] for i, l in zip(start_pos, seq_len)])
        else:
            values = np.empty((0, len(features)), dtype=np.float32)
        return (
            values,
            p2i["time"].to_numpy(dtype=np.float32),
            p2i["pid"].to_numpy(),
            features,
        )

    @classmethod
    def _read_features(cls, name: str) -> list[str]:
        with open(cls.base_dir / name / "features.yaml", "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def _read_index(cls, name: str) -> pd.DataFrame:
        return pd.read_csv(cls.base_dir / name / "p2i.csv")[["pid", "time"]]  # type: ignore


class ExpansionPack(TokenReader):

    base_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "expansion_packs"

    def __init__(self, name: str, memmap: bool = False):

        path = self.base_dir / name
        assert os.path.exists(path), FileNotFoundError(
            f"expansion pack {path} not found"
        )
        p2i = pd.read_csv(os.path.join(path, "p2i.csv"), index_col="pid")
        p2i = p2i[p2i["seq_len"] > 0]
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
        return p2i.loc[p2i["seq_len"] > 0, "pid"].unique()

    @classmethod
    def first_occurrence_times(cls, name: str, pids: np.ndarray) -> np.ndarray:
        pack = cls(name=name, memmap=True)
        result = np.full(len(pids), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in pack.start_pos:
                result[i] = pack.timesteps[pack.start_pos[pid]]
        return result

    @classmethod
    def catalog(cls) -> list[str]:
        """All expansion-pack names available under base_dir."""
        return list_subdirs(cls.base_dir, "tokenizer.yaml")


class MultimodalUKBReader(MultimodalReader):
    token_reader: UKBReader

    reader_cls = UKBReader
    biomarker_cls = Biomarker
    expansion_pack_cls = ExpansionPack

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

    def participants_with_event(self, pids: np.ndarray, event: str) -> np.ndarray:
        return self.token_reader.participants_with_event(pids, event)

    @classmethod
    def filter_participants_with_biomarkers(cls, pids, biomarkers, any=True):
        filter_list = [Biomarker.participants(b) for b in biomarkers]
        return filter_participants(pids, filter_list, any)

    @classmethod
    def filter_participants_with_expansion_packs(cls, pids, expansion_packs, any=True):
        filter_list = [ExpansionPack.participants(p) for p in expansion_packs]
        return filter_participants(pids, filter_list, any)

    @classmethod
    def filter_participants_with_modalities(cls, pids, biomarkers, expansion_packs):
        if biomarkers is not None:
            total = pids.size
            pids = cls.filter_participants_with_biomarkers(pids, biomarkers, any=True)
            print(f"{pids.size} / {total} pids (biomarker filter)")
        if expansion_packs is not None:
            total = pids.size
            pids = cls.filter_participants_with_expansion_packs(
                pids, expansion_packs, any=True
            )
            print(f"{pids.size} / {total} pids (expansion pack filter)")
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
