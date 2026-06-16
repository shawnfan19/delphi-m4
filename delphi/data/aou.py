# Register pandas extension dtypes for BigQuery-derived parquets (dbdate, dbtime).
import db_dtypes  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from cloudpathlib import AnyPath

from delphi.data.reader import (
    BiomarkerReader,
    ExpansionPackReader,
    MultimodalReader,
    TokenReader,
)
from delphi.env import DELPHI_DATA_READ as DELPHI_DATA_DIR

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


class Biomarker(BiomarkerReader):
    """AoU biomarker store. ``data.parquet`` already holds a 2-D measurement
    matrix (rows = measurements, cols = features), so ``_load`` reads the feature
    columns directly. All access logic lives on :class:`BiomarkerReader`."""

    base_dir = AnyPath(DELPHI_DATA_DIR) / "aou_uk" / "biomarkers"
    _marker = "data.parquet"

    def _load(self, name, memmap=False):  # memmap ignored: parquet has no memmap path
        path = self.base_dir / name / "data.parquet"
        assert path.exists(), FileNotFoundError(f"biomarker {path} not found")
        features = self._read_features(name)
        df = pd.read_parquet(
            path, columns=["person_id", "age_in_days"] + features
        ).sort_values(["person_id", "age_in_days"])
        return (
            df[features].to_numpy(dtype=np.float32),
            df["age_in_days"].to_numpy(dtype=np.float32),
            df["person_id"].to_numpy(),
            features,
        )

    @classmethod
    def _read_features(cls, name: str) -> list[str]:
        cols = pq.read_schema(str(cls.base_dir / name / "data.parquet")).names
        return _infer_features(cols)

    @classmethod
    def _read_index(cls, name: str) -> pd.DataFrame:
        df = pd.read_parquet(
            cls.base_dir / name / "data.parquet",
            columns=["person_id", "age_in_days"],
        )
        return df.rename(columns={"person_id": "pid", "age_in_days": "time"})


class ExpansionPack(ExpansionPackReader):
    """AoU expansion pack. data.parquet (person_id, age_in_days, token); indexed
    at load via np.unique. All access logic lives on ExpansionPackReader."""

    base_dir = AnyPath(DELPHI_DATA_DIR) / "aou_uk" / "expansion_packs"

    def _load(self, name, memmap=False):  # memmap ignored: parquet has no memmap path
        path = self.base_dir / name
        assert path.exists(), FileNotFoundError(f"expansion pack {path} not found")
        with open(path / "tokenizer.yaml", "r") as f:
            tokenizer = yaml.safe_load(f)
        df = pd.read_parquet(
            path / "data.parquet",
            columns=["person_id", "age_in_days", "token"],
        ).sort_values(["person_id", "age_in_days"])
        tokens = df["token"].to_numpy(dtype=np.uint32)
        timesteps = df["age_in_days"].to_numpy(dtype=np.float32)
        pids = df["person_id"].to_numpy()
        uniq, first_idx, counts = np.unique(pids, return_index=True, return_counts=True)
        start_pos = dict(zip(uniq, first_idx))
        seq_len = dict(zip(uniq, counts))
        return tokens, timesteps, start_pos, seq_len, tokenizer

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

    base_dir = AnyPath(DELPHI_DATA_DIR) / "aou_uk"
    biomarker_cls = Biomarker
    expansion_pack_cls = ExpansionPack

    bmi_keys = ["bmi_low", "bmi_mid", "bmi_high"]
    lifestyle_keys = bmi_keys
    sex_keys = ["female", "male"]
    FOLDS = ("val", "val_1", "val_2", "val_3", "val_4")

    def __init__(
        self,
        expansion_packs: list[str] | None = None,
        biomarkers: list[str] | dict[str, int] | None = None,
    ):
        bm_names, biomarker2idx = self._normalize_biomarkers(biomarkers)
        super().__init__(
            token_reader=self._load_token_reader(),
            expansion_packs={n: ExpansionPack(name=n) for n in expansion_packs or []},
            biomarkers={n: Biomarker(name=n) for n in bm_names},
            biomarker2idx=biomarker2idx,
        )

    @classmethod
    def _load_token_reader(cls) -> TokenReader:
        """Load the AoU main event stream (data.parquet) into a TokenReader."""
        with open(cls.base_dir / "tokenizer.yaml", "r") as f:
            tokenizer = yaml.safe_load(f)
        df = pd.read_parquet(
            cls.base_dir / "data.parquet",
            columns=["person_id", "age_in_days", "token"],
        ).sort_values(["person_id", "age_in_days"])
        tokens = df["token"].to_numpy(dtype=np.uint32)
        timesteps = df["age_in_days"].to_numpy(dtype=np.float32)
        pids = df["person_id"].to_numpy()
        uniq, first_idx, counts = np.unique(pids, return_index=True, return_counts=True)
        start_pos = pd.Series(first_idx, index=uniq)
        seq_len = pd.Series(counts, index=uniq)
        return TokenReader(tokens, timesteps, start_pos, seq_len, tokenizer)

    @classmethod
    def participants(cls, fold):
        pids = pd.read_parquet(cls.base_dir / "data.parquet", columns=["person_id"])[
            "person_id"
        ].unique()
        pids = np.sort(pids)
        if fold == "all":
            return pids
        if fold not in cls.FOLDS:
            raise ValueError(
                f"Unsupported fold {fold!r}; expected 'all' or one of {cls.FOLDS}"
            )
        return pids[cls.FOLDS.index(fold) :: len(cls.FOLDS)]

    @classmethod
    def first_biomarker_times(cls, pids: np.ndarray) -> np.ndarray:
        """Earliest measurement time across all biomarkers per participant.

        NaN where the participant has no biomarker measurements at all.
        """
        names = Biomarker.catalog()
        if not names:
            return np.full(len(pids), np.nan, dtype=np.float32)
        stack = np.stack(
            [Biomarker.first_occurrence_times(n, pids) for n in names], axis=0
        )
        return np.fmin.reduce(stack, axis=0)
