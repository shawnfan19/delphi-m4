from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from delphi.data.reader import TokenReader
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
