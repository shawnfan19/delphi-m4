from pathlib import Path

import numpy as np
import pandas as pd

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


def load_fids(fids: list[str | int]) -> pd.DataFrame:

    dfs = [load_fid(fid) for fid in fids]

    return pd.concat(dfs, axis=1)


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
