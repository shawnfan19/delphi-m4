from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


class UKBDatabase:
    """Reads UKB tabular data from a parquet conversion of the .tab file.

    The parquet at `parquet_path` is the authoritative source. The per-FID .txt
    extracts under `root/tab/` are retained for reference but unused.
    """

    VISITS = ["birth", "init_assess", "1st_repeat_assess", "img", "1st_repeat_img"]
    _ASSESS_INSTANCE = {
        "init_assess": 0,
        "1st_repeat_assess": 1,
        "img": 2,
        "1st_repeat_img": 3,
    }

    def __init__(self, root: str | Path, parquet_path: str | Path | None = None):
        self.root = Path(root)
        self.parquet_path = (
            Path(parquet_path) if parquet_path else self.root / "tab.parquet"
        )
        self._schema_cols: list[str] = pq.ParquetFile(
            self.parquet_path
        ).schema_arrow.names
        self._assess_age: pd.DataFrame | None = None
        self._long_assess_age: pd.Series | None = None

    def load_fid(self, fid: str | int) -> pd.DataFrame:
        fid = str(fid)
        cols = [c for c in self._schema_cols if c.startswith(f"f.{fid}.")]
        df = pq.read_table(self.parquet_path, columns=["f.eid", *cols]).to_pandas()
        df["f.eid"] = pd.to_numeric(df["f.eid"]).astype("int64")
        df = df.set_index("f.eid")
        return df

    def load_coding(self, scheme: int) -> pd.DataFrame:
        path = self.root / "coding" / f"{scheme}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path, sep="\t")

    def load_visit(self, fid: str | int, visit_idx: int = 0) -> dict:
        df = self.load_fid(fid)
        return df.iloc[:, visit_idx].to_dict()

    def month_of_birth(self) -> pd.DataFrame:
        mob = pd.read_csv(
            self.root / "year_and_month_of_birth.txt", sep="\t", index_col="eid"
        )
        mob["year_month"] = pd.to_datetime(mob["year_month"], format="%Y%m")
        return mob

    def assessment_age(self) -> pd.DataFrame:
        if self._assess_age is None:
            mob = self.month_of_birth()
            rename_map = {f"f.53.{i}.0": v for v, i in self._ASSESS_INSTANCE.items()}
            assess_date = self.load_fid(53).rename(columns=rename_map)
            assess_date["birth"] = mob["year_month"]

            assess_age = pd.DataFrame(
                index=assess_date.index, columns=self.VISITS, dtype=float
            )
            for col in self.VISITS:
                dt = pd.to_datetime(
                    assess_date[col], format="%Y-%m-%d", errors="coerce"
                )
                assess_age[col] = (dt - mob["year_month"]).dt.days.astype(float)
            self._assess_age = assess_age
        return self._assess_age

    def long_assessment_age(self) -> pd.Series:
        if self._long_assess_age is None:
            self._long_assess_age = index_by_visit(self.assessment_age(), self.VISITS)
        return self._long_assess_age

    def load_biomarker_df(
        self,
        fids: list[int | str],
        visits: list[str],
        name_by_fid: dict[int | str, str] | None = None,
    ) -> pd.DataFrame:
        markers = []
        for fid in fids:
            df = self.load_fid(fid).apply(pd.to_numeric, errors="coerce")
            marker = index_by_visit(df, visits)
            marker.name = name_by_fid[fid] if name_by_fid else str(fid)
            markers.append(marker)
        return pd.concat(markers, axis=1)


def load_ukb_biomarker_fids(data_dir: Path) -> dict[str, int]:
    """Merge biomarker.yaml (nested ukb field) + biomarker_ukb/*.yaml (flat) into name->fid."""
    fids: dict[str, int] = {}
    with open(data_dir / "biomarker.yaml") as f:
        for name, entry in yaml.safe_load(f).items():
            fids[name] = entry["ukb"]
    for path in sorted((data_dir / "biomarker_ukb").glob("*.yaml")):
        with open(path) as f:
            entries = yaml.safe_load(f) or {}
        duplicates = set(entries) & set(fids)
        if duplicates:
            raise ValueError(
                f"{path.name}: name(s) defined elsewhere too: {duplicates}"
            )
        fids.update(entries)
    return fids


def index_by_visit(df: pd.DataFrame, visits: list[str]) -> pd.Series:
    return (
        df.set_axis(visits, axis=1)
        .rename_axis(index="pid", columns="visit")
        .stack(future_stack=True)
    )


def build_biomarker(
    biomarker_df: pd.DataFrame,
    features: list,
    odir: str | Path,
    time_series: pd.Series,
    data_dtype=np.float32,
):
    odir = Path(odir)
    odir.mkdir(parents=True, exist_ok=True)
    print(odir)
    print(f"\t - features: {features}")

    with open(odir / "features.yaml", "w") as f:
        yaml.dump(features, f, default_flow_style=False, sort_keys=False)

    time_np = time_series[biomarker_df.index].to_numpy().astype(np.float32)
    data_np = biomarker_df.to_numpy().astype(data_dtype)

    has_nan_time = np.isnan(time_np)
    has_nan_data = np.isnan(data_np).any(axis=1)
    is_valid = ~has_nan_time & ~has_nan_data
    print(f"\t - has NaN in time: {has_nan_time.sum()}")
    print(f"\t - has NaN in data: {has_nan_data.sum()}")
    print(f"\t - total remaining: {is_valid.sum()}")

    kept = biomarker_df.loc[is_valid].reset_index()
    n_visits = kept["pid"].value_counts().value_counts().to_dict()
    print(f"\t - histogram: {n_visits}")

    seq_len = data_np.shape[1]
    p2i = pd.DataFrame(
        {
            "pid": kept["pid"].to_numpy(np.int32),
            "visit": kept["visit"].to_numpy(str),
            "start_pos": np.arange(is_valid.sum(), dtype=np.int32) * seq_len,
            "seq_len": seq_len,
            "time": time_np[is_valid],
        }
    )

    data_np[is_valid].ravel().tofile(odir / "data.bin")
    p2i.to_csv(odir / "p2i.csv", index=False)
