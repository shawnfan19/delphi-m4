from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from delphi.data.reader import MultimodalReader, TokenReader
from delphi.env import DELPHI_DATA_READ as DELPHI_DATA_DIR


class AOUReader(TokenReader):
    base_dir = Path(DELPHI_DATA_DIR) / "aou_uk"
    bmi_keys = [
        "bmi_low",
        "bmi_mid",
        "bmi_high",
    ]
    lifestyle_keys = bmi_keys
    sex_keys = ["female", "male"]

    def __init__(self):

        tokenizer_path = self.base_dir / "tokenizer.yaml"
        with open(tokenizer_path, "r") as f:
            tokenizer = yaml.safe_load(f)

        df = pd.read_parquet(self.base_dir / "data.parquet")
        df = df.sort_values(["person_id", "age_in_days"])

        tokens = df["token"].to_numpy(dtype=np.uint32)
        timesteps = df["age_in_days"].to_numpy(dtype=np.float32)

        pids = df["person_id"].to_numpy()
        uniq, first_idx, counts = np.unique(pids, return_index=True, return_counts=True)
        start_pos = pd.Series(first_idx, index=uniq)
        seq_len = pd.Series(counts, index=uniq)

        super().__init__(tokens, timesteps, start_pos, seq_len, tokenizer)

    @classmethod
    def participants(cls, fold):
        if fold != "all":
            raise ValueError(
                f"Unsupported fold {fold!r}; only 'all' is supported for now"
            )
        pids = pd.read_parquet(cls.base_dir / "data.parquet", columns=["person_id"])[
            "person_id"
        ].unique()
        return np.sort(pids)

    def is_female(self, pids: np.ndarray) -> np.ndarray:
        female_token = self.tokenizer["female"]
        out = np.zeros(len(pids), dtype=bool)
        for i, pid in enumerate(pids):
            start = self.start_pos[int(pid)]
            length = self.seq_len[int(pid)]
            out[i] = (self.tokens[start : start + length] == female_token).any()
        return out


METADATA_SUFFIXES = (
    "_raw_value",
    "_unit_id",
    "_unit_name",
    "_concept_id",
    "_concept_name",
)


def _infer_features(columns) -> list[str]:
    cols = set(columns)
    return [c for c in columns if all(f"{c}{suf}" in cols for suf in METADATA_SUFFIXES)]


class AOUBiomarker:

    base_dir = Path(DELPHI_DATA_DIR) / "aou_uk" / "biomarkers"

    def __init__(self, name: str, first_time_only: bool = True):
        path = self.base_dir / name / "data.parquet"
        assert path.exists(), FileNotFoundError(f"biomarker {path} not found")
        self.path = path

        df = pd.read_parquet(path).sort_values(["person_id", "age_in_days"])
        features = _infer_features(df.columns)

        self.features = features
        self.feat2idx = {f: i for i, f in enumerate(features)}
        self.n_features = len(features)

        self.data = df[features].to_numpy(dtype=np.float32)
        self.time_steps = df["age_in_days"].to_numpy(dtype=np.float32)
        self.pids = df["person_id"].to_numpy()

        uniq, first_idx, counts = np.unique(
            self.pids, return_index=True, return_counts=True
        )
        self.pid2idx = dict(zip(uniq, first_idx))
        self.pid2cnt = dict(zip(uniq, counts))

        self.first_time_only = first_time_only

    @classmethod
    def input_size(cls, name: str) -> int:
        import pyarrow.parquet as pq

        cols = pq.read_schema(cls.base_dir / name / "data.parquet").names
        return len(_infer_features(cols))

    @classmethod
    def participants(cls, name: str) -> np.ndarray:
        df = pd.read_parquet(
            cls.base_dir / name / "data.parquet", columns=["person_id"]
        )
        return df["person_id"].unique()

    @classmethod
    def first_occurrence_times(cls, name: str, pids: np.ndarray) -> np.ndarray:
        df = pd.read_parquet(
            cls.base_dir / name / "data.parquet",
            columns=["person_id", "age_in_days"],
        ).sort_values(["person_id", "age_in_days"])
        first = df.groupby("person_id")["age_in_days"].first()
        result = np.full(len(pids), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in first.index:
                result[i] = first.loc[pid]
        return result

    def __repr__(self):
        return f"AOUBiomarker(path={self.path}, n_features={self.n_features})"

    def __getitem__(
        self, pid: int
    ) -> tuple[None | list[np.ndarray], None | np.ndarray]:

        if pid not in self.pid2idx:
            return None, None

        i = self.pid2idx[pid]
        n = self.pid2cnt[pid]
        if self.first_time_only:
            return [self.data[i]], self.time_steps[i : i + 1]
        pid_data = [self.data[j] for j in range(i, i + n)]
        return pid_data, self.time_steps[i : i + n]

    def to_array(self, subjects):
        data, subs = list(), list()
        include = np.isin(self.pids, subjects)
        feat_data = self.data[include]
        pids = self.pids[include]
        seen = set()
        for j, pid in enumerate(pids):
            if self.first_time_only:
                if pid in seen:
                    continue
                seen.add(pid)
            data.append(feat_data[j])
            subs.append(pid)
        return np.stack(data, axis=0), np.array(subs)

    def stats(self, subjects: np.ndarray):
        data, _ = self.to_array(subjects)
        return np.mean(data, axis=0), np.std(data, axis=0)


class AOUExpansionPack(TokenReader):

    base_dir = Path(DELPHI_DATA_DIR) / "aou_uk" / "expansion_packs"

    def __init__(self, name: str):
        path = self.base_dir / name
        assert path.exists(), FileNotFoundError(f"expansion pack {path} not found")

        tokenizer_path = path / "tokenizer.yaml"
        with open(tokenizer_path, "r") as f:
            tokenizer = yaml.safe_load(f)

        df = pd.read_parquet(path / "data.parquet").sort_values(
            ["person_id", "age_in_days"]
        )

        tokens = df["token"].to_numpy(dtype=np.uint32)
        timesteps = df["age_in_days"].to_numpy(dtype=np.float32)

        pids = df["person_id"].to_numpy()
        uniq, first_idx, counts = np.unique(pids, return_index=True, return_counts=True)
        self.pids = uniq
        start_pos = pd.Series(first_idx, index=uniq)
        seq_len = pd.Series(counts, index=uniq)

        super().__init__(tokens, timesteps, start_pos, seq_len, tokenizer)

    @classmethod
    def participants(cls, name: str) -> np.ndarray:
        df = pd.read_parquet(
            cls.base_dir / name / "data.parquet", columns=["person_id"]
        )
        return df["person_id"].unique()

    @classmethod
    def first_occurrence_times(cls, name: str, pids: np.ndarray) -> np.ndarray:
        df = pd.read_parquet(
            cls.base_dir / name / "data.parquet",
            columns=["person_id", "age_in_days"],
        ).sort_values(["person_id", "age_in_days"])
        first = df.groupby("person_id")["age_in_days"].first()
        result = np.full(len(pids), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in first.index:
                result[i] = first.loc[pid]
        return result


class MultimodalAOUReader(MultimodalReader):
    token_reader: AOUReader

    bmi_keys = AOUReader.bmi_keys
    lifestyle_keys = AOUReader.lifestyle_keys
    sex_keys = AOUReader.sex_keys

    def __init__(
        self,
        expansion_packs: list[str] | None = None,
        biomarkers: list[str] | dict[str, int] | None = None,
    ):
        bm_names, biomarker2idx = self._normalize_biomarkers(biomarkers)
        super().__init__(
            token_reader=AOUReader(),
            expansion_packs={
                n: AOUExpansionPack(name=n) for n in expansion_packs or []
            },
            biomarkers={n: AOUBiomarker(name=n) for n in bm_names},
            biomarker2idx=biomarker2idx,
        )

    @classmethod
    def participants(cls, fold):
        return AOUReader.participants(fold)

    def is_female(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.is_female(pids)

    def event_times(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.event_times(pids)

    def exit_times(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.exit_times(pids)
