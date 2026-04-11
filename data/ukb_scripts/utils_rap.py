import os
from pathlib import Path
import dxdata  # type: ignore
import numpy as np
import pandas as pd
import yaml

import pyspark.sql.functions as F

from delphi.env import DELPHI_DATA_DIR


project = os.getenv("DX_PROJECT_CONTEXT_ID")
assert project is not None
record = (
    os.popen("dx find data --type Dataset --delimiter ',' | awk -F ',' '{print $5}'")
    .read()
    .rstrip()
)
DATASET_ID = project + ":" + record
print(DATASET_ID)

engine = dxdata.connect(dialect="hive+pyspark")
dataset = dxdata.load_dataset(id=DATASET_ID)

pheno = dataset["participant"]
field_eid = pheno.find_field(name="eid")

VISITS = ["birth", "init_assess", "1st_repeat_assess", "img", "1st_repeat_img"]


def all_ukb_participants() -> np.ndarray:

    participant_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "participants"
    participants = np.fromfile(participant_dir / "all.bin", dtype=np.uint32)

    return participants


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


def regex_search(fid):

    fields = list(pheno.find_fields(name_regex=f".*p{fid}_.*"))
    if len(fields) == 0:
        fields = list(pheno.find_fields(name_regex=f".*p{fid}.*"))
        
    return sorted(fields, key=lambda field: field.name)


def load_fids(fids: list[str | int]):
    """Returns a PySpark DataFrame natively, avoiding immediate collection to Pandas."""
    fields = list()
    for fid in fids:
        fields.extend(regex_search(fid))
    fields.append(field_eid)
    
    df = pheno.retrieve_fields(
        engine=engine, 
        fields=fields,
    )
    return df


_mob_cache = None
def month_of_birth():
    global _mob_cache
    if _mob_cache is None:
        field_birth_year = pheno.find_field(title="Year of birth")  
        field_birth_month = pheno.find_field(title="Month of birth") 
        
        mob = pheno.retrieve_fields(
            engine=engine, 
            fields=[field_eid, field_birth_year, field_birth_month],
            coding_values="replace"
        )
        
        mob = mob.select(
            F.col("eid"),
            F.col(field_birth_year.name).alias("year"),
            F.col(field_birth_month.name).alias("month_str")
        )
        
        # Parse strings like "January" natively into a Spark Date representing the 1st of that month
        mob = mob.withColumn(
            "year_month",
            F.to_date(F.concat_ws("-", F.col("year"), F.col("month_str"), F.lit("1")), "yyyy-MMMM-d")
        )
        
        mob = mob.withColumn("day", F.lit(1))
        mob = mob.withColumn("month", F.month("year_month"))
        
        _mob_cache = mob.cache() # Caches dataframe in Spark executor memory
    return _mob_cache


_assess_age_cache = None
def assessment_age():
    global _assess_age_cache
    if _assess_age_cache is None:
        mob = month_of_birth()
        assess_date = load_fids(fids=[53]) # Date of attending assessment center
        
        df = assess_date.join(mob, on="eid", how="left")
        
        mapping = {
            "init_assess": "p53_i0",
            "1st_repeat_assess": "p53_i1",
            "img": "p53_i2",
            "1st_repeat_img": "p53_i3",
        }
        
        select_exprs = [F.col("eid")]
        
        for col in VISITS:
            if col == "birth":
                select_exprs.append(F.lit(0.0).alias("birth"))
                continue

            p53_col = mapping[col]
            # Calculate the difference in days using Spark's datediff
            diff_days = F.datediff(F.to_date(F.col(p53_col)), F.col("year_month")).cast("double")
            select_exprs.append(diff_days.alias(col))
                
        _assess_age_cache = df.select(*select_exprs).cache()
    return _assess_age_cache


_long_assess_age_cache = None
def _long_assessment_age():
    global _long_assess_age_cache
    if _long_assess_age_cache is None:
        assess_age = assessment_age()
        
        stack_args = list()
        for visit in VISITS:
            stack_args.append(F.lit(visit))
            stack_args.append(F.col(visit))      
        _long_assess_age_cache = assess_age.select(
            "eid", F.stack(F.lit(len(VISITS)), *stack_args).alias("visit", "time")
        ).cache()
    return _long_assess_age_cache


def load_biomarker_df(fids: list[int | str], visits: list[str]):
    """
    Loads FIDs and performs wide-to-long conversion using Spark natively.
    Returns a PySpark DataFrame with columns: eid, visit, <fid1>, <fid2>...
    """
    df = load_fids(fids)
    
    fid_columns = {}
    for fid in fids:
        fields = regex_search(fid)
        field_names = [field.name for field in fields]
        assert len(field_names) == len(visits), f"Visits length mismatch for FID {fid}"
        fid_columns[fid] = field_names

    num_visits = len(visits)
    stack_args = list()
    for i, visit_name in enumerate(visits):
        stack_args.append(F.lit(visit_name)) # Label
        for fid in fids:
            col_name = fid_columns[fid][i]
            stack_args.append(F.col(col_name)) # Value
    
    alias_names = ["visit"] + [str(f) for f in fids]
    
    long_df = df.select(
        "eid", 
        F.stack(F.lit(num_visits), *stack_args).alias(*alias_names)
    )

    return long_df


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

    p2i = pd.DataFrame(
        {
            "pid": subjects,
            "start_pos": np.cumsum(count_np) - count_np,
            "seq_len": count_np,
        }
    )
    p2i = p2i.set_index("pid")

    odir = Path(odir) / expansion_pack
    os.makedirs(odir, exist_ok=True)
    p2i.to_csv(odir / "p2i.csv")
    token_np.astype(np.uint32).tofile(odir / "data.bin")
    time_np = time_np.astype(np.uint32)
    print(
        f"\t - time points from {time_np.min() / 365.25} to {time_np.max() / 365.25}"
    )
    time_np.tofile(odir / "time.bin")

    with open(odir / "tokenizer.yaml", "w") as f:
        yaml.dump(
            tokenizer,
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def build_spark_biomarker(
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
        
    df = biomarker_df.withColumnRenamed("eid", "pid")
    
    # Optional: Cache the raw dataframe to make our sequential .count() logs extremely fast
    df = df.cache()
    
    ukb_subjects = all_ukb_participants()
    valid_pids_df = df.sparkSession.createDataFrame(pd.DataFrame({"pid": ukb_subjects}))
    
    count_before = df.count()
    # A broadcast Inner Join efficiently drops anyone not in the valid cohort
    biomarker_df = df.join(F.broadcast(valid_pids_df), on="pid", how="inner")
    count_after_ukb = df.count()
    print(f"\t - not found in Delphi cohort: {count_before - count_after_ukb}")

    time_df = _long_assessment_age()
    time_df = time_df.withColumnRenamed("eid", "pid")
    df = df.join(time_df, on=["pid", "visit"], how="left")
    df = df.dropna(subset=["time"])
    count_after_time = df.count()
    print(f"\t - has NaN in time: {count_after_ukb - count_after_time}")
    
    
    meta_cols = {"pid", "visit", "time"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    df = df.dropna(subset=feature_cols)
    count_after_data = df.count()
    print(f"\t - has NaN in data: {count_after_time - count_after_data}")
    print(f"\t - total remaining: {count_after_data}")
    
    # First, get the number of visits per PID, explicitly naming the result "num_visits"
    visit_counts = df.groupBy("pid").agg(F.count("*").alias("num_visits"))
    
    # Next, group by "num_visits" to get how many people had that many visits
    histogram_rows = visit_counts.groupBy("num_visits").agg(F.count("*").alias("frequency")).collect()
    
    # Safely build the dictionary
    histogram = {row["num_visits"]: row["frequency"] for row in histogram_rows}
    print(f"\t - histogram: {histogram}")

    # =========================================================
    # THE HANDOFF: Spark to Local Memory
    # Spark doesn't guarantee row order. We MUST order by pid 
    # and visit to ensure the physical binary sequences align.
    # =========================================================
    df_final = df.orderBy("pid", "visit").toPandas()
    
    # Free up cluster memory
    df.unpersist()
    
    subjects = df_final["pid"].to_numpy().astype(np.int32)
    visits = df_final["visit"].to_numpy().astype(str)
    time_np = df_final["time"].to_numpy().astype(np.float32)
    data_np = df_final[feature_cols].to_numpy().astype(data_dtype)
    seq_len = data_np.shape[1]
    
    p2i = pd.DataFrame.from_dict(
        data={
            "pid": subjects,
            "visit": visits,
            "start_pos": (np.cumsum(np.full(len(df_final), seq_len)) - seq_len).astype(
                np.int32
            ),
            "seq_len": seq_len,
            "time": time_np,
        }
    )

    data_np.ravel().astype(np.float32).tofile(odir / "data.bin")
    p2i.to_csv(odir / "p2i.csv", index=False)

    
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
    time_df = _long_assessment_age()
    time_df = time_df.withColumnRenamed("eid", "pid")
    time_df = time_df.toPandas()
    time_df = time_df.set_index(["pid", "visit"])
    time_series = time_df["time"]
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