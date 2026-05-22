import pprint
from functools import cached_property
from typing import Protocol

import numpy as np

from delphi.data.utils import sort_by_time, update_tokenizer

RESERVED_MOD_IDX = 2  # 0 = padding, 1 = event tokens


class _Biomarker(Protocol):
    """Structural type for biomarker instances composed by MultimodalReader."""

    def __getitem__(
        self, pid: int
    ) -> tuple[list[np.ndarray] | None, np.ndarray | None]: ...


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

    def event_times(self, pids: np.ndarray) -> np.ndarray:
        """N by (max_token_id+1) array of first-occurrence times; NaN where a token never occurs."""
        n_cols = max(self.tokenizer.values()) + 1
        out = np.full((len(pids), n_cols), np.nan, dtype=np.float32)
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
        expansion_packs: dict[str, TokenReader] | None = None,
        biomarkers: dict[str, _Biomarker] | None = None,
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
