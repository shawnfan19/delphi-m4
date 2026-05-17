# +
from typing import TypedDict
import yaml
from sqlalchemy import (
    create_engine,
    select,
    and_,
    or_,
    case,
    func,
    literal_column,
    Table,
    MetaData,
)
from sqlalchemy.orm import declarative_base
from google.cloud import storage

from utils import Client, WORKSPACE_CDR, DATA_BUCKET, PROJECT_ID
# -

from pathlib import Path
import os
CWD = Path(os.getcwd())
CWD


# +
class Biomarker(TypedDict):
    """
    id:
        - 3020460 # crp
    unit:
        8751: null # milligram per liter
        8840: 1e-2 # milligram per deciliter
        4122414: null # mg/L

    """
    id: list[int]
    unit: dict[int, float | None]


engine = create_engine(f'bigquery://{WORKSPACE_CDR}')
metadata = MetaData()
measurement = Table('measurement', metadata, schema=WORKSPACE_CDR, autoload_with=engine)
concept = Table('concept', metadata, schema=WORKSPACE_CDR, autoload_with=engine)
person = Table('person', metadata, schema=WORKSPACE_CDR, autoload_with=engine)


def create_unit_conversion_case(
    unit_id2spec: dict
):
    """
    Generates a reusable SQLAlchemy CASE expression for unit conversion.
    
    Instead of a string, this returns a composable SQL expression object.
    """
    whens = {}
    val = measurement.c.value_as_number
    for unit_id, spec in unit_id2spec.items():
        if spec is None:
            continue
        if isinstance(spec, dict):
            factor = spec.get("factor", 1.0)
            offset = spec.get("offset", 0.0)
            whens[unit_id] = val * factor + offset
        else:
            whens[unit_id] = val * spec
    if not whens:
        # If no conversions, just return the original value
        return measurement.c.value_as_number
    return case(
        whens,
        value=measurement.c.unit_concept_id,
        else_=measurement.c.value_as_number
    )


def create_biomarker_query(name: str, biomarker: Biomarker):
    """
    Builds the entire query using SQLAlchemy's composable expressions.
    """
    measurement_concept_ids = biomarker["id"]
    unit_concept_ids = list(biomarker["unit"].keys())
    
    unit_conversion_case = create_unit_conversion_case(biomarker["unit"])

    row_number = func.row_number().over(
        partition_by=[
            measurement.c.person_id,
            measurement.c.visit_occurrence_id,
            func.date(measurement.c.measurement_date)
        ],
        order_by=measurement.c.measurement_date.desc()
    )

    biomarker_cte = (
        select(
            measurement.c.person_id,
            measurement.c.measurement_date,
            func.DATE_DIFF(
                func.date(measurement.c.measurement_date),
                func.date(person.c.birth_datetime),
                literal_column("DAY")
            ).label("age_in_days"),
            measurement.c.measurement_concept_id,
            measurement.c.value_as_number,
            unit_conversion_case.label("standardized_value"),
            concept.c.concept_name.label("unit_name"),
            row_number.label("rn")
        )
        .join(person, measurement.c.person_id == person.c.person_id)
        .join(concept, measurement.c.unit_concept_id == concept.c.concept_id)
        .where(
            and_(
                measurement.c.value_as_number.is_not(None),
                and_(
                    measurement.c.measurement_concept_id.in_(measurement_concept_ids),
                    measurement.c.unit_concept_id.in_(unit_concept_ids)
                )
            )
        )
        .cte("biomarker")
    )

    final_query = (
        select(
            biomarker_cte.c.person_id,
            biomarker_cte.c.measurement_date,
            biomarker_cte.c.standardized_value.label(name), # Use the aliased column from the CTE
            biomarker_cte.c.unit_name
        )
        .where(biomarker_cte.c.rn == 1)
        .order_by(
            biomarker_cte.c.person_id,
            biomarker_cte.c.measurement_date
        )
    )
    
    return final_query


def create_biomarker_name_case(biomarkers: dict[str: Biomarker]):
    whens = list()
    for name, biomarker in biomarkers.items():
        condition = measurement.c.measurement_concept_id.in_(
            biomarker["id"]
        )
        whens.append((condition, name))

    return case(*whens)


def create_biomarker_panel_query(biomarkers: dict[str: Biomarker]):

    wheres = list()
    for biomarker in biomarkers.values():
        wheres.append(
            and_(
                measurement.c.measurement_concept_id.in_(
                    biomarker["id"]
                ),
                measurement.c.unit_concept_id.in_(
                    list(biomarker["unit"].keys())
                )
            )
        )

    conversion_whens = list()
    for name, biomarker in biomarkers.items():
        unit_conversion_case = create_unit_conversion_case(biomarker['unit'])
        condition = measurement.c.measurement_concept_id.in_(
            biomarker["id"]
        )
        conversion_whens.append((condition, unit_conversion_case))

    master_unit_conversion_case = case(
        *conversion_whens,
        else_=measurement.c.value_as_number  # Fallback for any other measurement
    )

    biomarker_name = create_biomarker_name_case(biomarkers)

    row_number = func.row_number().over(
        partition_by=[
            measurement.c.person_id,
            measurement.c.visit_occurrence_id,
            func.date(measurement.c.measurement_date),
            biomarker_name
        ],
        order_by=measurement.c.measurement_date.desc()
    )

    unit_concept = concept.alias("unit_concept")
    meas_concept = concept.alias("meas_concept")
    biomarker_cte = (
        select(
            measurement.c.person_id,
            measurement.c.measurement_date,
            measurement.c.visit_occurrence_id,
            func.DATE_DIFF(
                func.date(measurement.c.measurement_date),
                func.date(person.c.birth_datetime),
                literal_column("DAY")
            ).label("age_in_days"),
            measurement.c.measurement_concept_id,
            meas_concept.c.concept_name.label("concept_name"),
            measurement.c.value_as_number,
            master_unit_conversion_case.label("standardized_value"),
            measurement.c.unit_concept_id,
            unit_concept.c.concept_name.label("unit_name"),
            biomarker_name.label("biomarker"),
            row_number.label("rn")
        )
        .join(person, measurement.c.person_id == person.c.person_id)
        .join(unit_concept, measurement.c.unit_concept_id == unit_concept.c.concept_id)
        .join(meas_concept, measurement.c.measurement_concept_id == meas_concept.c.concept_id)
        .where(
            and_(
                measurement.c.value_as_number.is_not(None),
                or_(*wheres)
            )
        )
        .cte("biomarker")
    )

    pivot_targets = {
        "": biomarker_cte.c.standardized_value,
        "_raw_value": biomarker_cte.c.value_as_number,
        "_unit_id": biomarker_cte.c.unit_concept_id,
        "_unit_name": biomarker_cte.c.unit_name,
        "_concept_id": biomarker_cte.c.measurement_concept_id,
        "_concept_name": biomarker_cte.c.concept_name,
    }
    biomarker_cols = []
    for name in biomarkers.keys():
        for suffix, col in pivot_targets.items():
            label = f"{name}{suffix}"
            biomarker_cols.append(
                func.max(
                    case(
                        (biomarker_cte.c.biomarker == name, col),
                    )
                ).label(label)
            )

    group_cols = [
        biomarker_cte.c.person_id,
        biomarker_cte.c.visit_occurrence_id,
        biomarker_cte.c.measurement_date,
        biomarker_cte.c.age_in_days,
    ]
    final_query = (
        select(
            *group_cols,
            *biomarker_cols
        )
        .where(biomarker_cte.c.rn == 1)
        .group_by(*group_cols)
        .order_by(
            biomarker_cte.c.person_id,
            biomarker_cte.c.age_in_days
        )
    )

    return final_query


# -
client = Client(dataset=WORKSPACE_CDR)



with open(CWD.parent / "biomarker.yaml", "r") as f:
    _biomarker_dict = yaml.safe_load(f)
biomarker_dict = dict()
for key, value in _biomarker_dict.items():
    if "aou" in value.keys():
        biomarker_dict[key] = value["aou"]

biomarker_dict

# get availability of all biomarkers 
biomarker_count = dict()
for name, biomarker in biomarker_dict.items():
    ids = biomarker["id"]
    ids_str = ", ".join(str(i) for i in ids)
    
    q = f"""
    SELECT concept_id, domain_id, name, est_count, rollup_count
    FROM   `cb_criteria`
    WHERE  domain_id = 'MEASUREMENT'
      AND  is_standard = 1
      AND  is_selectable = 1
      AND  is_group = 0
      AND  concept_id in ({ids_str});
    """
    
    df = client.run(q)
    est_ct = df["est_count"].iloc[0]
    biomarker_count[name] = int(est_ct)
print(biomarker_count)

biomarker_count = {k: int(v) for k, v in biomarker_count.items()}
with open(CWD / "biomarker_availability.yaml", "w") as f:
    yaml.dump(biomarker_count, f)

with open(CWD.parent / "aou_panel.yaml", "r") as f:
    panels = yaml.safe_load(f)
panels

# +
storage_client = storage.Client()
bucket = storage_client.bucket(DATA_BUCKET)

for panel_name, biomarker_list in panels.items():
    if len(biomarker_list) == 1:
        q = create_biomarker_query(
            biomarker_list[0],
            biomarker_dict[biomarker_list[0]]
        )
    else:
        q = create_biomarker_panel_query(
            {name: biomarker_dict[name] for name in biomarker_list}
        )

    # To see the generated SQL (the equivalent of a "dry run"):
    # Use the BigQuery dialect to compile it into Google Standard SQL
    compiled_q = q.compile(
        dialect=engine.dialect,
        compile_kwargs={"literal_binds": True}
    )

    df = client.run(str(compiled_q))
    # odir = Path.home() / f"workspace/data/aou_uk/biomarkers/{panel_name}"
    # os.makedirs(odir, exist_ok=True)
    df.to_parquet(
        f"gs://{DATA_BUCKET}/aou_uk/biomarkers/{panel_name}/data.parquet", index=False
    )

    missing_counts = df[biomarker_list].isna().sum().to_dict()
    blob = bucket.blob(f"aou_uk/biomarkers/{panel_name}/missing_counts.yaml")
    blob.upload_from_string(
        yaml.dump(missing_counts),
        content_type="text/yaml",
    )
    # with open(odir / "missing_counts.yaml", "w") as f:
    #     yaml.dump(missing_counts, f)
# -

DATA_BUCKET


