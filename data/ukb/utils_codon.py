import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from delphi.env import DELPHI_DATA_DIR

_raw_dir = Path(DELPHI_DATA_DIR) / "ukb"
_tab_dir = _raw_dir / "tab"
_pheno = {}
for file in _tab_dir.rglob("*"):
    if file.is_file():
        key = file.stem
        key = key.split("_")[-1]
        _pheno[key] = str(file)


VISITS = ["birth", "init_assess", "1st_repeat_assess", "img", "1st_repeat_img"]


def load_fid(fid: str | int, preload: None | pd.DataFrame = None) -> pd.DataFrame:

    return pd.read_csv(_pheno[str(fid)], delimiter="\t", index_col="f.eid")


def load_biomarker_df(fids: list, visits: list[str]) -> pd.DataFrame:
    "load fids and perform wide-to-long conversion"
    markers = []
    for fid in fids:
        marker = load_fid(str(fid))
        marker = index_by_visit(df=marker, visits=visits)
        marker.name = str(fid)
        markers.append(marker)
    long_df = pd.concat(markers, axis=1)

    return long_df


def load_coding(scheme: int) -> pd.DataFrame:

    coding_path = _raw_dir / "coding" / f"{str(scheme)}.txt"
    if not coding_path.exists():
        raise FileNotFoundError(f"Coding file {coding_path} does not exist.")

    return pd.read_csv(coding_path, sep="\t")


def load_visit(fid: str, visit_idx: int = 0) -> dict:
    """
    return a dictionary that maps participant IDs to a measurement from a given visit specified by visit_idx
    """

    df = load_fid(fid=fid)
    assert visit_idx < df.shape[1], "visit index out of bounds"

    return df.iloc[:, visit_idx].to_dict()


def month_of_birth() -> pd.DataFrame:

    mob = pd.read_csv(
        _raw_dir / "year_and_month_of_birth.txt", sep="\t", index_col="eid"
    )
    mob["year_month"] = pd.to_datetime(mob["year_month"], format="%Y%m")

    return mob


_assess_age_cache = None


def assessment_age():

    global _assess_age_cache
    mob = month_of_birth()

    if _assess_age_cache is None:
        assess_date = load_fid(fid=53)
        assess_date = assess_date.rename(
            columns={
                "f.53.0.0": "init_assess",
                "f.53.1.0": "1st_repeat_assess",
                "f.53.2.0": "img",
                "f.53.3.0": "1st_repeat_img",
            }
        )
        assess_date["birth"] = mob["year_month"]

        assess_age = pd.DataFrame(columns=np.array(VISITS), index=assess_date.index)
        for col in VISITS:
            assess_date[col] = pd.to_datetime(assess_date[col], format="%Y-%m-%d")
            assess_age[col] = (assess_date[col] - mob["year_month"]).dt.days.astype(
                float
            )
        _assess_age_cache = assess_age

    return _assess_age_cache


_long_assess_age_cache = None


def _long_assessment_age() -> pd.Series:

    global _long_assess_age_cache
    if _long_assess_age_cache is None:
        assess_age = assessment_age()
        long_assess_age = index_by_visit(assess_age, visits=VISITS)
        _long_assess_age_cache = long_assess_age

    return _long_assess_age_cache


def index_by_visit(df: pd.DataFrame, visits: list[str]) -> pd.Series:

    n = df.shape[0]
    l = df.shape[1]
    assert l == len(
        visits
    ), "Number of visits does not match number of columns in DataFrame"
    vals = np.concatenate([df[col].to_numpy() for col in df.columns], axis=0)
    visit_types = np.repeat(np.array(visits), n)
    subjects = np.tile(df.index.to_numpy(), l)

    return pd.Series(
        data=vals,
        index=pd.MultiIndex.from_arrays(
            [subjects, visit_types], names=["pid", "visit"]
        ),
    )


def build_biomarker(
    biomarker_df: pd.DataFrame,
    features: list,
    odir: str | Path,
    data_dtype=np.float32,
):

    odir = Path(odir)
    os.makedirs(odir, exist_ok=True)
    print(f"{odir}")

    print(f"\t - features: {features}")
    with open(odir / "features.yaml", "w") as f:
        yaml.dump(
            features,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    subjects = biomarker_df.reset_index()["pid"].to_numpy().astype(np.int32)
    visits = biomarker_df.reset_index()["visit"].to_numpy().astype(str)

    time_series = _long_assessment_age()
    time_np = time_series[biomarker_df.index].to_numpy().astype(np.float32)
    has_nan_time = np.isnan(time_np)
    print(f"\t - has NaN in time: {has_nan_time.sum()}")
    is_valid = ~has_nan_time

    data_np = biomarker_df.to_numpy().astype(data_dtype)
    has_nan_data = np.isnan(biomarker_df.values).any(axis=1)
    print(f"\t - has NaN in data: {has_nan_data.sum()}")
    is_valid *= ~has_nan_data

    print(f"\t - total remaining: {is_valid.sum()}")
    histogram = (
        biomarker_df.loc[is_valid]
        .reset_index()["pid"]
        .value_counts()
        .value_counts()
        .to_dict()
    )
    print(f"\t - histogram: {histogram}")

    data_np = data_np[is_valid]
    time_np = time_np[is_valid]
    subjects = subjects[is_valid]
    visits = visits[is_valid]

    seq_len = data_np.shape[1]
    p2i = pd.DataFrame.from_dict(
        data={
            "pid": subjects,
            "visit": visits,
            "start_pos": (np.cumsum(np.full(is_valid.sum(), seq_len)) - seq_len).astype(
                np.int32
            ),
            "seq_len": seq_len,
            "time": time_np,
        }
    )

    data_np.ravel().astype(np.float32).tofile(odir / "data.bin")
    p2i.to_csv(odir / "p2i.csv", index=False)
