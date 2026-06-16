import abc
import pprint
from collections.abc import Mapping
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from delphi.data.utils import list_subdirs, sort_by_time, update_tokenizer

RESERVED_MOD_IDX = 2  # 0 = padding, 1 = event tokens


class BiomarkerReader(abc.ABC):
    """Abstract biomarker store: shared access logic + a per-dataset storage seam.

    Concrete subclasses (one per dataset, all named ``Biomarker`` because only
    one dataset is ever live in a given environment) implement *only* the
    storage adapter: the ``base_dir`` / ``_marker`` class attributes and the
    ``_load`` / ``_read_features`` / ``_read_index`` hooks. Everything else —
    per-pid access, first-occurrence vectors, stats, and the disk-only catalog
    queries — lives here once.

    Canonical in-memory layout (each ``_load`` normalizes its storage to it):
        data:  (n_measurements, n_features) float32, rows grouped by pid then
               ordered by time.
        times: (n_measurements,) float32, aligned row-for-row to ``data``.
        pid2idx / pid2cnt: pid -> first row / number of rows in ``data``.
    """

    # Set by subclasses: base_dir is a pathlib.Path (UKB) or cloudpathlib.AnyPath
    # (AoU); _marker is the filename marking a biomarker dir.
    base_dir: ClassVar[Any]
    _marker: ClassVar[str]

    # ---- storage seam: the only per-dataset code ----------------------------
    @abc.abstractmethod
    def _load(
        self, name: str, memmap: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Return (values[N, n_features], times[N], pids[N], features) sorted by (pid, time).

        ``memmap`` is a UKB-only storage hint; datasets without a memmap path
        accept and ignore it.
        """

    @classmethod
    @abc.abstractmethod
    def _read_features(cls, name: str) -> list[str]:
        """Feature names for ``name``, read from disk without building an instance."""

    @classmethod
    @abc.abstractmethod
    def _read_index(cls, name: str) -> pd.DataFrame:
        """A (pid, time) table for ``name`` (no feature payload), read from disk."""

    # ---- shared construction: index derivation lives once -------------------
    def __init__(self, name: str, memmap: bool = False):
        self.name = name
        self.data, self.times, pids, self.features = self._load(name, memmap=memmap)
        self.feat2idx = {feat: i for i, feat in enumerate(self.features)}
        self.n_features = len(self.features)
        uniq, first_idx, counts = np.unique(pids, return_index=True, return_counts=True)
        self.pid2idx = dict(zip(uniq, first_idx))
        self.pid2cnt = dict(zip(uniq, counts))

    # ---- shared per-instance access -----------------------------------------
    def __getitem__(
        self, pid: int
    ) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
        if pid not in self.pid2idx:
            return None, None
        i, n = self.pid2idx[pid], self.pid2cnt[pid]
        return list(self.data[i : i + n]), self.times[i : i + n]

    def to_array(self, subjects) -> np.ndarray:
        """First-occurrence feature vector per subject, aligned to ``subjects``.

        Returns an (len(subjects), n_features) array; rows for subjects with no
        measurement are NaN. pid2idx points at each pid's first row, which —
        because rows are time-ordered within a pid — is the earliest measurement.
        """
        out = np.full((len(subjects), self.n_features), np.nan, dtype=np.float32)
        for k, pid in enumerate(subjects):
            j = self.pid2idx.get(int(pid))
            if j is not None:  # absent (note: j may be 0, so test `is None`)
                out[k] = self.data[j]
        return out

    def stats(self, subjects: np.ndarray):
        data = self.to_array(subjects)
        return np.nanmean(data, axis=0), np.nanstd(data, axis=0)

    def __repr__(self):
        return (
            f"{type(self).__name__}(name={self.name!r}, n_features={self.n_features})"
        )

    # ---- shared disk-only catalog queries (no instance needed) --------------
    @classmethod
    def catalog(cls) -> list[str]:
        """All biomarker names available under base_dir."""
        return list_subdirs(cls.base_dir, cls._marker)

    @classmethod
    def input_size(cls, name: str) -> int:
        return len(cls._read_features(name))

    @classmethod
    def participants(cls, name: str) -> np.ndarray:
        return cls._read_index(name)["pid"].unique()  # type: ignore

    @classmethod
    def first_occurrence_times(cls, name: str, pids: np.ndarray) -> np.ndarray:
        idx = cls._read_index(name).sort_values(["pid", "time"])
        first = idx.groupby("pid")["time"].first()
        result = np.full(len(pids), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in first.index:
                result[i] = first.loc[pid]
        return result


class TokenReader:
    """Base for token-sequence readers — generic per-pid queries over (tokens, timesteps)."""

    def __init__(self, tokens, timesteps, start_pos, seq_len, tokenizer):
        self.tokens = tokens
        self.timesteps = timesteps
        self.start_pos = start_pos
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)

    @cached_property
    def detokenizer(self):
        return {v: k for k, v in self.tokenizer.items()}

    def __getitem__(self, pid: int):

        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x_pid = self.tokens[i : i + l].astype(np.uint32)
        t_pid = self.timesteps[i : i + l].astype(np.float32)

        return x_pid, t_pid


class ExpansionPackReader(abc.ABC):
    """Abstract expansion-pack store: a supplementary token stream merged into
    the main vocabulary.

    Composes a :class:`TokenReader` for per-pid (token, time) access and exposes
    only the slicing surface the composer reads. Concrete subclasses (both named
    ``ExpansionPack``, one per dataset) implement only the storage seam: the
    ``base_dir`` class attribute, the ``_load`` hook, and the disk-only
    ``participants`` / ``first_occurrence_times`` queries (whose backends — UKB
    ``time.bin`` vs AoU parquet — genuinely differ, so they stay abstract).
    """

    base_dir: ClassVar[Any]
    _marker: ClassVar[str] = "tokenizer.yaml"

    @abc.abstractmethod
    def _load(
        self, name: str, memmap: bool = False
    ) -> tuple[np.ndarray, np.ndarray, dict, dict, dict]:
        """Return (tokens, timesteps, start_pos, seq_len, tokenizer) for the pack."""

    def __init__(self, name: str, memmap: bool = False):
        self.name = name
        tokens, timesteps, start_pos, seq_len, tokenizer = self._load(
            name, memmap=memmap
        )
        self.reader = TokenReader(tokens, timesteps, start_pos, seq_len, tokenizer)
        self.pids = np.array(list(start_pos))

    # ---- delegated slicing surface (what the composer reads) ----------------
    def __getitem__(self, pid: int):
        return self.reader[pid]

    @property
    def start_pos(self):
        return self.reader.start_pos

    @property
    def seq_len(self):
        return self.reader.seq_len

    @property
    def tokenizer(self):
        return self.reader.tokenizer

    # ---- disk-only catalog queries (no full instance needed) ----------------
    @classmethod
    def catalog(cls) -> list[str]:
        """All expansion-pack names available under base_dir."""
        return list_subdirs(cls.base_dir, cls._marker)

    @classmethod
    @abc.abstractmethod
    def participants(cls, name: str) -> np.ndarray:
        """Pids present in the pack ``name``, read from disk."""

    @classmethod
    @abc.abstractmethod
    def first_occurrence_times(cls, name: str, pids: np.ndarray) -> np.ndarray:
        """Earliest pack-token time per pid, aligned to ``pids`` (NaN if absent)."""


class MultimodalReader:
    """Composes a base TokenReader + expansion packs + biomarkers.

    Pure composer. Subclasses are responsible for constructing the components
    (token reader, packs, biomarkers) and passing them in; the base only
    stores them and assembles them in __getitem__. Dataset-specific concerns
    (file loading, classmethods like participants, methods like is_female)
    live on the subclass.
    """

    def __init__(
        self,
        token_reader: TokenReader,
        expansion_packs: Mapping[str, ExpansionPackReader] | None = None,
        biomarkers: Mapping[str, BiomarkerReader] | None = None,
        biomarker2idx: dict[str, int] | None = None,
    ):
        self.token_reader = token_reader
        self.base_tokenizer = token_reader.tokenizer.copy()
        self.tokenizer = self.base_tokenizer.copy()

        self.expansion_packs = dict()
        self.expansion_offset = dict()
        for name in sorted(expansion_packs or {}):
            pack = (expansion_packs or {})[name]
            self.expansion_packs[name] = pack
            self.tokenizer, offset = update_tokenizer(
                base_tokenizer=self.tokenizer,
                add_tokenizer=pack.tokenizer,
            )
            self.expansion_offset[name] = offset
        self.vocab_size = len(self.tokenizer)

        if biomarker2idx is None:
            self.biomarker2idx = {
                name.lower(): i + RESERVED_MOD_IDX
                for i, name in enumerate(sorted(biomarkers or {}))
            }
        else:
            self.biomarker2idx = {k.lower(): v for k, v in biomarker2idx.items()}
            bad = [k for k, v in self.biomarker2idx.items() if v < RESERVED_MOD_IDX]
            if bad:
                raise ValueError(
                    f"biomarker indices must be >= {RESERVED_MOD_IDX} "
                    f"(0=padding, 1=event token); got {bad}"
                )
        self.biomarkers = {k.lower(): v for k, v in (biomarkers or {}).items()}

    @staticmethod
    def _normalize_biomarkers(
        spec: list[str] | dict[str, int] | None,
    ) -> tuple[list[str], dict[str, int] | None]:
        """Parse the public biomarkers arg into (names, optional explicit idx mapping)."""
        if spec is None:
            return [], None
        if isinstance(spec, list):
            return spec, None
        if isinstance(spec, dict):
            return list(spec.keys()), spec
        raise ValueError(
            f"biomarkers must be list[str] or dict[str, int], "
            f"got {type(spec).__name__}"
        )

    @cached_property
    def detokenizer(self):
        return {v: k for k, v in self.tokenizer.items()}

    @property
    def expansion_tokens(self):
        return list(set(self.tokenizer.values()) - set(self.base_tokenizer.values()))

    def describe(self) -> None:
        print(f"{type(self).__name__}:")
        config = {
            "expansion_packs": sorted(self.expansion_packs.keys()),
            "biomarkers": sorted(self.biomarkers),
        }
        pprint.pp(config)

    # ---- full-sequence trajectory queries over the main token stream --------
    def event_times(self, pids: np.ndarray) -> np.ndarray:
        """N by (max main-stream token id + 1) array of first-occurrence times;
        NaN where a token never occurs."""
        tr = self.token_reader
        n_cols = max(tr.tokenizer.values()) + 1
        out = np.full((len(pids), n_cols), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            start = tr.start_pos[int(pid)]
            length = tr.seq_len[int(pid)]
            x = tr.tokens[start : start + length]
            t = tr.timesteps[start : start + length].astype(np.float32)
            uniq, first_idx = np.unique(x, return_index=True)
            out[i, uniq] = t[first_idx]
        return out

    def exit_times(self, pids: np.ndarray) -> np.ndarray:
        """N array of last token times (exit / censoring time)."""
        tr = self.token_reader
        out = np.empty(len(pids), dtype=np.float32)
        for i, pid in enumerate(pids):
            start = tr.start_pos[int(pid)]
            length = tr.seq_len[int(pid)]
            out[i] = tr.timesteps[start + length - 1]
        return out

    def participants_with_event(self, pids: np.ndarray, event: str) -> np.ndarray:
        tr = self.token_reader
        token = tr.tokenizer[event]
        pids_with_event = list()
        for pid in pids:
            start = tr.start_pos[int(pid)]
            length = tr.seq_len[int(pid)]
            if token in tr.tokens[start : start + length]:
                pids_with_event.append(pid)
        return np.array(pids_with_event)

    def __getitem__(self, pid: int):

        x, t = self.token_reader[pid]
        x_lst, t_lst = [x], [t]
        for name, expansion_pack in self.expansion_packs.items():
            if pid not in expansion_pack.start_pos:
                continue
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
            mod_m = np.full_like(mod_t, fill_value=self.biomarker2idx[modality])
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
