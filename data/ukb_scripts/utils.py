import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from delphi import DAYS_PER_YEAR
from delphi.env import IN_RAP

if IN_RAP:
    from utils_rap import (
        all_ukb_participants,
        assessment_age,
        load_fid,
        month_of_birth
    )
else:
    from utils_codon import (
        all_ukb_participants,
        assessment_age,
        load_fid,
        month_of_birth
    )


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


def load_biomarker_df(fids: list, visits: list[str]) -> pd.DataFrame:

    markers = []
    for fid in fids:
        marker = load_fid(str(fid))
        marker = index_by_visit(df=marker, visits=visits)
        marker.name = str(fid)
        markers.append(marker)
    long_df = pd.concat(markers, axis=1)

    return long_df


def load_visit(fid: str, visit_idx: int = 0) -> dict:
    """
    return a dictionary that maps participant IDs to a measurement from a given visit specified by visit_idx
    """

    df = load_fid(fid=fid)
    assert visit_idx < df.shape[1], "visit index out of bounds"

    return df.iloc[:, visit_idx].to_dict()


def init_expansion_pack_p2i():

    ukb_subjects = all_ukb_participants()
    p2i = pd.DataFrame(
        {
            "pid": ukb_subjects,
            "start_pos": 0,
            "seq_len": 0,
        }
    )
    p2i = p2i.set_index("pid")

    return p2i


def build_expansion_pack(
    token_np: np.ndarray,
    time_np: np.ndarray,
    count_np: np.ndarray,
    subjects: np.ndarray,
    tokenizer: dict,
    odir: str | os.PathLike,
    expansion_pack: str,
):
    print(expansion_pack)
    assert token_np.size == time_np.size
    assert count_np.sum() == token_np.size
    assert subjects.size == count_np.size
    print(f"\t - total tokens: {token_np.size}")
    print(f"\t - subjects: {subjects.size}")
    print(f"\t - avg tokens per subject: {count_np.mean()}")
    print(f"\t - max tokens per subject: {count_np.max()}")
    print(f"\t - vocab size: {len(tokenizer)}")

    p2i = init_expansion_pack_p2i()
    p2i.loc[subjects, "seq_len"] = count_np
    p2i.loc[subjects, "start_pos"] = np.cumsum(count_np) - count_np
    p2i.loc[p2i["seq_len"] == 0, "start_pos"] = 0

    odir = Path(odir) / expansion_pack
    os.makedirs(odir, exist_ok=True)
    p2i.to_csv(odir / "p2i.csv")
    token_np.astype(np.uint32).tofile(odir / "data.bin")
    time_np = time_np.astype(np.uint32)
    print(
        f"\t - time points from {time_np.min() / DAYS_PER_YEAR} to {time_np.max() / DAYS_PER_YEAR}"
    )
    time_np.tofile(odir / "time.bin")

    with open(odir / "tokenizer.yaml", "w") as f:
        yaml.dump(
            tokenizer,
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def build_biomarker(
    biomarker_df: pd.DataFrame,
    time_series: pd.Series,
    odir,
    str,
    biomarker: str,
    data_dtype=np.float32,
):

    print(biomarker)
    odir = Path(odir) / biomarker
    os.makedirs(odir, exist_ok=True)

    features = biomarker_df.columns.tolist()
    print(features)
    with open(odir / "features.yaml", "w") as f:
        yaml.dump(
            features,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    subjects = biomarker_df.reset_index()["pid"].to_numpy().astype(np.int32)
    visits = biomarker_df.reset_index()["visit"].to_numpy().astype(str)

    ukb_subjects = all_ukb_participants()
    not_in_ukb_subjects = ~np.isin(subjects, ukb_subjects)
    print(f"\t - not found in Delphi cohort: {not_in_ukb_subjects.sum()}")
    is_valid = ~not_in_ukb_subjects

    time_np = time_series[biomarker_df.index].to_numpy().astype(np.float32)
    has_nan_time = np.isnan(time_np)
    print(f"\t - has NaN in time: {has_nan_time.sum()}")
    is_valid *= ~has_nan_time

    data_np = biomarker_df.to_numpy().astype(data_dtype)
    has_nan_data = biomarker_df.isna().any(axis=1)
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

    miss_subjects = list(set(ukb_subjects) - set(subjects))
    miss_p2i = pd.DataFrame.from_dict(
        data={
            "pid": miss_subjects,
            "visit": "none",
            "start_pos": 0,
            "seq_len": 0,
            "time": -1e4,
        }
    )

    p2i = pd.concat([p2i, miss_p2i], axis=0)

    data_np.ravel().astype(np.float32).tofile(odir / "data.bin")
    p2i.to_csv(odir / "p2i.csv", index=False)
