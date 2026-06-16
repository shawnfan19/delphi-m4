# Register pandas extension dtypes for BigQuery-derived parquets (dbdate, dbtime).
import db_dtypes  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from cloudpathlib import AnyPath

from delphi.data.reader import BiomarkerReader, MultimodalReader, TokenReader
from delphi.data.utils import filter_participants, list_subdirs
from delphi.env import DELPHI_DATA_READ as DELPHI_DATA_DIR


class AOUReader(TokenReader):
    base_dir = AnyPath(DELPHI_DATA_DIR) / "aou_uk"
    bmi_keys = [
        "bmi_low",
        "bmi_mid",
        "bmi_high",
    ]
    lifestyle_keys = bmi_keys
    sex_keys = ["female", "male"]
    FOLDS = ("val", "val_1", "val_2", "val_3", "val_4")

    def __init__(self):

        tokenizer_path = self.base_dir / "tokenizer.yaml"
        with open(tokenizer_path, "r") as f:
            tokenizer = yaml.safe_load(f)

        df = pd.read_parquet(
            self.base_dir / "data.parquet",
            columns=["person_id", "age_in_days", "token"],
        ).sort_values(["person_id", "age_in_days"])

        tokens = df["token"].to_numpy(dtype=np.uint32)
        timesteps = df["age_in_days"].to_numpy(dtype=np.float32)

        pids = df["person_id"].to_numpy()
        uniq, first_idx, counts = np.unique(pids, return_index=True, return_counts=True)
        start_pos = pd.Series(first_idx, index=uniq)
        seq_len = pd.Series(counts, index=uniq)

        super().__init__(tokens, timesteps, start_pos, seq_len, tokenizer)

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


class AOUExpansionPack(TokenReader):

    base_dir = AnyPath(DELPHI_DATA_DIR) / "aou_uk" / "expansion_packs"

    def __init__(self, name: str):
        path = self.base_dir / name
        assert path.exists(), FileNotFoundError(f"expansion pack {path} not found")

        tokenizer_path = path / "tokenizer.yaml"
        with open(tokenizer_path, "r") as f:
            tokenizer = yaml.safe_load(f)

        df = pd.read_parquet(
            path / "data.parquet",
            columns=["person_id", "age_in_days", "token"],
        ).sort_values(["person_id", "age_in_days"])

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

    @classmethod
    def catalog(cls) -> list[str]:
        """All expansion-pack names available under base_dir."""
        return list_subdirs(cls.base_dir, "tokenizer.yaml")


class MultimodalAOUReader(MultimodalReader):
    token_reader: AOUReader

    reader_cls = AOUReader
    biomarker_cls = Biomarker
    expansion_pack_cls = AOUExpansionPack

    bmi_keys = AOUReader.bmi_keys
    lifestyle_keys = AOUReader.lifestyle_keys
    sex_keys = AOUReader.sex_keys
    FOLDS = AOUReader.FOLDS

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
            biomarkers={n: Biomarker(name=n) for n in bm_names},
            biomarker2idx=biomarker2idx,
        )

    @classmethod
    def participants(cls, fold):
        return AOUReader.participants(fold)

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

    def is_female(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.is_female(pids)

    def event_times(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.event_times(pids)

    def exit_times(self, pids: np.ndarray) -> np.ndarray:
        return self.token_reader.exit_times(pids)

    @classmethod
    def filter_participants_with_biomarkers(cls, pids, biomarkers, any=True):
        filter_list = [Biomarker.participants(b) for b in biomarkers]
        return filter_participants(pids, filter_list, any)

    @classmethod
    def filter_participants_with_expansion_packs(cls, pids, expansion_packs, any=True):
        filter_list = [AOUExpansionPack.participants(p) for p in expansion_packs]
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
