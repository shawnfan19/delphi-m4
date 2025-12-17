import os

import dxdata  # type: ignore
import numpy as np
import pandas as pd

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

_mob_cache = None


def month_of_birth() -> pd.DataFrame:

    global _mob_cache
    if _mob_cache is None:
        field_birth_year = pheno.find_field(title="Year of birth")  # 34
        field_birth_month = pheno.find_field(title="Month of birth")  # 52
        mob = pheno.retrieve_fields(
            engine=engine, fields=[field_eid, field_birth_year, field_birth_month],
            coding_values="replace"
        ).toPandas()
        mob = mob.rename(columns={"p34": "year", "p52": "month"})
        mob = mob.set_index("eid")
        _mob_cache = mob
        _mob_cache["day"] = 1
        _mob_cache["month"] = pd.to_datetime(_mob_cache["month"], format="%B").dt.month
        _mob_cache["year_month"] = pd.to_datetime(_mob_cache)

    return _mob_cache


def regex_search(fid):

    fields = list(pheno.find_fields(name_regex=f".*p{fid}_.*"))
    if len(fields) == 0:
        fields = list(pheno.find_fields(name_regex=f".*p{fid}.*"))
        
    return fields


def load_fid(fid: str | int, preload: None | pd.DataFrame = None) -> pd.DataFrame:
    fields = regex_search(fid)
    if preload is None:
        fields.append(field_eid)
        fid_df = pheno.retrieve_fields(
            engine=engine, fields=fields,
            # coding_values="replace"
        ).toPandas()
        fid_df = fid_df.set_index("eid")
    else:
        fields = [field.name for field in fields]
        fid_df = preload[fields]
    return fid_df


def load_fids(fids: list[str | int]) -> pd.DataFrame:
    fields = list()
    for fid in fids:
        fields.extend(regex_search(fid))
    fields.append(field_eid)
    df = pheno.retrieve_fields(
        engine=engine, fields=fields,
        # coding_values="replace"
    ).toPandas()
    df = df.set_index("eid")
    return df


_assess_age_cache = None


def assessment_age():

    global _assess_age_cache
    mob = month_of_birth()

    if _assess_age_cache is None:
        assess_date = load_fid(fid=53)
        assess_date = assess_date.rename(
            columns={
                "p53_i0": "init_assess",
                "p53_i1": "1st_repeat_assess",
                "p53_i2": "img",
                "p53_i3": "1st_repeat_img",
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
