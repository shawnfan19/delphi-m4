from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REQUIRED_COLUMNS = ("person_id", "age_in_days", "token")
REQUIRED_TOKENS = ("padding", "no_event", "male", "female")
MAX_AGE_IN_DAYS = 100 * 365.25


def required_files_exist(dataset_dir: Path) -> bool:
    return (dataset_dir / "data.parquet").exists() and (
        dataset_dir / "tokenizer.yaml"
    ).exists()


def has_required_columns(df: pd.DataFrame) -> bool:
    return set(REQUIRED_COLUMNS).issubset(df.columns)


def no_nan_in_required_columns(df: pd.DataFrame) -> bool:
    return not df[list(REQUIRED_COLUMNS)].isna().any().any()


def tokens_within_range(tokens: np.ndarray, tokenizer: dict) -> bool:
    min_token = int(np.min(tokens))
    max_token = int(np.max(tokens))
    return (
        min_token >= 2
        and min_token >= min(tokenizer.values())
        and max_token <= max(tokenizer.values())
    )


def tokenizer_contiguous(tokenizer: dict) -> bool:
    vals = list(tokenizer.values())
    return len(vals) == len(set(vals)) and max(vals) == len(vals) - 1


def tokenizer_contains_required_pairs(tokenizer: dict) -> bool:
    return all(t in tokenizer for t in REQUIRED_TOKENS)


def age_in_days_in_human_range(ages: np.ndarray) -> bool:
    return bool(np.all(ages >= 0) and np.all(ages <= MAX_AGE_IN_DAYS))


def no_duplicate_event_triples(df: pd.DataFrame) -> bool:
    return not df.duplicated(subset=list(REQUIRED_COLUMNS)).any()


def parquet_sorted_by(df: pd.DataFrame, cols: list[str]) -> bool:
    original = df[cols].reset_index(drop=True)
    sorted_df = df[cols].sort_values(by=cols).reset_index(drop=True)
    return original.equals(sorted_df)


def test_data(dataset_dir):

    assert required_files_exist(dataset_dir)

    with open(dataset_dir / "tokenizer.yaml", "r") as f:
        tokenizer = yaml.safe_load(f)

    df = pd.read_parquet(dataset_dir / "data.parquet", columns=list(REQUIRED_COLUMNS))
    tokens = df["token"].to_numpy()
    ages = df["age_in_days"].to_numpy()

    assert has_required_columns(df=df)
    assert no_nan_in_required_columns(df=df)
    assert tokens_within_range(tokens=tokens, tokenizer=tokenizer)
    assert tokenizer_contiguous(tokenizer=tokenizer)
    assert tokenizer_contains_required_pairs(tokenizer=tokenizer)
    assert age_in_days_in_human_range(ages=ages)
    assert no_duplicate_event_triples(df=df)
    assert parquet_sorted_by(df=df, cols=["person_id", "age_in_days"])
