from functools import cached_property

import numpy as np


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
