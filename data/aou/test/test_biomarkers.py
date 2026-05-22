from pathlib import Path

import numpy as np
import pandas as pd
from test_data import (
    age_in_days_in_human_range,
    parquet_sorted_by,
)

from delphi.data.aou import _infer_features

BASE_COLUMNS = ("person_id", "age_in_days")


def data_parquet_exists(panel_path: Path) -> bool:
    return (panel_path / "data.parquet").exists()


def has_base_columns(df: pd.DataFrame) -> bool:
    return set(BASE_COLUMNS).issubset(df.columns)


def at_least_one_feature(features: list[str]) -> bool:
    return len(features) > 0


def no_nan_in_features_and_base(df: pd.DataFrame, features: list[str]) -> bool:
    cols = list(BASE_COLUMNS) + list(features)
    return not df[cols].isna().any().any()


def n_features_matches_config(
    features: list[str], panel_name: str, panel_config: dict
) -> bool:
    declared = panel_config.get(panel_name)
    if declared is None:
        raise AssertionError(f"panel {panel_name!r} not found in data/panel/aou.yaml")
    return len(features) == len(declared)


def test_biomarkers(dataset_dir, panel, panel_config):

    panel_path = dataset_dir / "biomarkers" / panel
    assert data_parquet_exists(panel_path=panel_path)

    df = pd.read_parquet(panel_path / "data.parquet")
    features = _infer_features(df.columns)

    assert has_base_columns(df=df)
    assert at_least_one_feature(features=features)
    assert no_nan_in_features_and_base(df=df, features=features)
    assert age_in_days_in_human_range(ages=df["age_in_days"].to_numpy())
    assert parquet_sorted_by(df=df, cols=list(BASE_COLUMNS))
    assert n_features_matches_config(
        features=features, panel_name=panel, panel_config=panel_config
    )
