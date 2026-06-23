import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from cloudpathlib import AnyPath

from delphi.data.reader import (
    BiomarkerReader,
    ExpansionPackReader,
    MultimodalReader,
    TokenReader,
)
from delphi.env import DELPHI_DATA_READ as DELPHI_DATA_DIR

NO_EVENT_TOKEN = 1


class Biomarker(BiomarkerReader):
    """UKB biomarker store. Flat ``data.bin`` (float32) + ``p2i.csv`` (pid, time,
    start_pos, seq_len); ``_load`` gathers the ragged-flat rows into the 2-D
    canonical layout. All access logic lives on :class:`BiomarkerReader`."""

    base_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers"
    _marker = "data.bin"

    def _load(self, name):
        path = self.base_dir / name
        assert os.path.exists(path), FileNotFoundError(f"biomarker {path} not found")
        flat = np.fromfile(os.path.join(path, "data.bin"), dtype=np.float32)

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


class ExpansionPack(ExpansionPackReader):
    """UKB expansion pack. Flat data.bin/time.bin (uint32) + p2i.csv
    (pid, start_pos, seq_len; pack times live in time.bin, not p2i)."""

    base_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "expansion_packs"

    def _load(self, name):
        path = self.base_dir / name
        assert os.path.exists(path), FileNotFoundError(
            f"expansion pack {path} not found"
        )
        p2i = pd.read_csv(os.path.join(path, "p2i.csv"), index_col="pid")
        p2i = p2i[p2i["seq_len"] > 0]
        start_pos = p2i["start_pos"].to_dict()
        seq_len = p2i["seq_len"].to_dict()
        tokens = np.fromfile(os.path.join(path, "data.bin"), dtype=np.uint32)
        timesteps = np.fromfile(os.path.join(path, "time.bin"), dtype=np.uint32)
        with open(os.path.join(path, "tokenizer.yaml"), "r") as f:
            tokenizer = yaml.safe_load(f)
        return tokens, timesteps, start_pos, seq_len, tokenizer

    @classmethod
    def participants(cls, name: str) -> np.ndarray:
        p2i = pd.read_csv(cls.base_dir / name / "p2i.csv")
        return p2i.loc[p2i["seq_len"] > 0, "pid"].unique()

    @classmethod
    def first_occurrence_times(cls, name: str, pids: np.ndarray) -> np.ndarray:
        pack = cls(name=name)
        result = np.full(len(pids), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in pack.start_pos:
                result[i] = pack.reader.timesteps[pack.start_pos[pid]]
        return result


class MultimodalUKBReader(MultimodalReader):

    base_dir = AnyPath(DELPHI_DATA_DIR) / "ukb_real_data"
    biomarker_cls = Biomarker
    expansion_pack_cls = ExpansionPack
    named_times = frozenset({"recruitment"})

    bmi_keys = ["bmi_low", "bmi_mid", "bmi_high"]
    smoking_keys = ["smoking_low", "smoking_mid", "smoking_high"]
    alcohol_keys = ["alcohol_low", "alcohol_mid", "alcohol_high"]
    lifestyle_keys = bmi_keys + smoking_keys + alcohol_keys
    sex_keys = ["female", "male"]

    @classmethod
    def _load_token_reader(cls) -> TokenReader:
        """Load the UKB main event stream (data.bin/time.bin + p2i.csv) into a TokenReader."""
        with open(cls.base_dir / "tokenizer.yaml", "r") as f:
            tokenizer = yaml.safe_load(f)
        p2i = pd.read_csv(cls.base_dir / "p2i.csv", index_col="pid")
        start_pos = p2i["start_pos"].to_dict()
        seq_len = p2i["seq_len"].to_dict()
        tokens = np.fromfile(cls.base_dir / "data.bin", dtype=np.uint32)
        timesteps = np.fromfile(cls.base_dir / "time.bin", dtype=np.uint32)
        return TokenReader(tokens, timesteps, start_pos, seq_len, tokenizer)

    @classmethod
    def participants(cls, fold):
        if fold == "all":
            return pd.read_csv(cls.base_dir / "p2i.csv", usecols=["pid"])[
                "pid"
            ].to_numpy(dtype=np.uint32)
        return np.fromfile(
            cls.base_dir / "participants" / f"{fold}_fold.bin", dtype=np.uint32
        )

    def recruitment_times(self, pids: np.ndarray) -> np.ndarray:
        """Earliest lifestyle-token time per pid (UKB recruitment proxy); NaN if none.

        The main stream is time-ordered per pid, so the earliest lifestyle-token
        time equals its first-occurrence time.
        """
        tr = self.token_reader
        lifestyle_tokens = np.array([tr.tokenizer[e] for e in self.lifestyle_keys])
        out = np.full(len(pids), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            start = tr.start_pos[int(pid)]
            length = tr.seq_len[int(pid)]
            x = tr.tokens[start : start + length]
            t = tr.timesteps[start : start + length].astype(np.float32)
            mask = np.isin(x, lifestyle_tokens)
            if mask.any():
                out[i] = t[mask].min()
        return out
