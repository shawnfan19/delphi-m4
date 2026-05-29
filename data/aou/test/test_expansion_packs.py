import numpy as np
import pandas as pd
import yaml
from test_data import (
    REQUIRED_COLUMNS,
    age_in_days_in_human_range,
    has_required_columns,
    no_duplicate_event_triples,
    no_nan_in_required_columns,
    parquet_sorted_by,
)


def data_parquet_exists(pack_path) -> bool:
    return (pack_path / "data.parquet").exists()


def tokenizer_exists(pack_path) -> bool:
    return (pack_path / "tokenizer.yaml").exists()


def tokens_within_range(tokens: np.ndarray, tokenizer: dict) -> bool:
    # Expansion-pack tokens index from 1 (no padding / no_event), so — unlike the
    # disease tokens in test_data — there is no `>= 2` floor.
    return int(np.min(tokens)) >= min(tokenizer.values()) and int(
        np.max(tokens)
    ) <= max(tokenizer.values())


def tokenizer_contiguous(tokenizer: dict) -> bool:
    # Expansion packs are 1-indexed: values run 1..N, so max == len (contrast
    # test_data.tokenizer_contiguous, where the 0-indexed disease vocab has max == len - 1).
    vals = list(tokenizer.values())
    return len(vals) == len(set(vals)) and min(vals) == 1 and max(vals) == len(vals)


def test_expansion_pack(dataset_dir, expansion_pack):

    pack_path = dataset_dir / "expansion_packs" / expansion_pack
    assert data_parquet_exists(pack_path=pack_path)
    assert tokenizer_exists(pack_path=pack_path)

    with open(pack_path / "tokenizer.yaml", "r") as f:
        tokenizer = yaml.safe_load(f)

    df = pd.read_parquet(pack_path / "data.parquet", columns=list(REQUIRED_COLUMNS))
    tokens = df["token"].to_numpy()
    ages = df["age_in_days"].to_numpy()

    assert has_required_columns(df=df)
    assert no_nan_in_required_columns(df=df)
    assert tokens_within_range(tokens=tokens, tokenizer=tokenizer)
    assert tokenizer_contiguous(tokenizer=tokenizer)
    assert age_in_days_in_human_range(ages=ages)
    assert no_duplicate_event_triples(df=df)
    assert parquet_sorted_by(df=df, cols=["person_id", "age_in_days"])
