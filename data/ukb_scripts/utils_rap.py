import os
from datetime import datetime

import dxdata
import numpy as np
import pandas as pd

project = os.getenv("DX_PROJECT_CONTEXT_ID")
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


def all_ukb_participants() -> np.ndarray:
    field_eid = pheno.find_field(name="eid")
    eid = pheno.retrieve_fields(
        engine=engine, fields=[field_eid], coding_values="replace"
    ).toPandas()
    eid = eid["eid"].values.astype(np.int64)
    return eid


def month_of_birth() -> pd.DataFrame:
    field_birth_year = pheno.find_field(title="Year of birth")  # 34
    birth_year = pheno.retrieve_fields(
        engine=engine, fields=[field_birth_year], coding_values="replace"
    ).toPandas()
    field_birth_month = pheno.find_field(title="Month of birth")  # 52
    birth_month = pheno.retrieve_fields(
        engine=engine, fields=[field_birth_month], coding_values="replace"
    ).toPandas()
    mob = pd.concat((birth_year, birth_month), axis=1)
    mob = mob.rename(columns={"p34": "year", "p52": "month"})
    mob["day"] = 1
    mob["month"] = pd.to_datetime(mob["month"], format="%B").dt.month
    mob["year_month"] = pd.to_datetime(mob)
    return mob


def load_fid(fid: str | int) -> pd.DataFrame:
    fields = list(pheno.find_fields(name_regex=f".*p{fid}_.*"))
    fid_df = pheno.retrieve_fields(
        engine=engine, fields=fields, coding_values="replace"
    ).toPandas()
    return fid_df


def assessment_age(visits: list[str]):

    mob = month_of_birth()
    assess_date = load_fid(fid=53)
    assess_date = assess_date.rename(
        columns={
            "p53_i0": "init_assess",
            "p53_i1": "1st_repeat_assess",
            "p53_i2": "img",
            "p53_i3": "1st_repeat_img",
        }
    )
    assert set(visits).issubset(set(assess_date.columns))
    age_at_visits = pd.DataFrame(columns=visits, index=assess_date.index)
    for col in visits:
        assess_date[col] = pd.to_datetime(assess_date[col], format="%Y-%m-%d")
        age_at_visits[col] = (assess_date[col] - mob["year_month"]).dt.days.astype(
            float
        )

    return age_at_visits
