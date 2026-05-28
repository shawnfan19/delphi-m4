# +
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from utils import DATA_BUCKET, WORKSPACE_CDR, Client, upload_yaml

# -

try:
    _CORE_DIR = Path(__file__).resolve().parent
except NameError:  # Jupyter cell execution
    _CORE_DIR = Path(os.getcwd())
TOKENIZER_PATH = _CORE_DIR.parent / "ukb" / "dictionary" / "tokenizer.yaml"

client = Client(dataset=WORKSPACE_CDR)

client.list_columns("condition_occurrence")

client.list_columns("condition_occurrence_ext")

client.list_rows("condition_occurrence")

client.list_columns("person")

client.value_counts("person", "sex_at_birth_concept_id")

client.list_columns("measurement_ext")

client.unique("measurement_ext", "src_id")

client.list_columns("measurement")

# # disease tokens

query = """
SELECT
    co.person_id,
    co.condition_concept_id AS concept_id,
    -- This brings in the cleanly formatted ICD code if the source was ICD
    c.concept_code AS concept_code,
    c.vocabulary_id AS vocab_id,
    DATE_DIFF(DATE(co.condition_start_date), DATE(p.birth_datetime), DAY) AS age_in_days
FROM
    `condition_occurrence` co
-- Join the source concept ID back to the concept table
LEFT JOIN
    `concept` c
    ON co.condition_source_concept_id = c.concept_id
INNER JOIN
    `person` p
    ON p.person_id = co.person_id
ORDER BY person_id, age_in_days;
"""
client.dry_run(query)

# +
import time

t0 = time.time()
df = client.run(query)
t = time.time() - t0
print(f"query in {t} minutes")
# -

df.head()

df.shape

df.vocab_id.value_counts()

# ## ICD tokens

icd10_df = df[df.vocab_id == "ICD10CM"].copy()
icd10_df.head()

assert icd10_df.concept_code.isna().sum() == 0

icd10_df.shape

icd10_df = icd10_df.drop_duplicates(subset=["person_id", "concept_code"])
icd10_df.shape

icd10_df["icd_code"] = icd10_df.concept_code.str.split(".", expand=True)[0]

icd10_df.head()

# ## SNOMED tokens

snomed_df = df[~(df.vocab_id == "ICD10CM")].copy()

snomed_df.shape

snomed_df = snomed_df.drop_duplicates(subset=["person_id", "concept_id"])
snomed_df.shape

# ### get SNOMED2ICD mapping

q = f"""
WITH icd10 AS (
    SELECT
        concept_id,
        SUBSTR(concept_code, 1, 3) AS icd_code
    FROM concept
    WHERE vocabulary_id = 'ICD10CM'
),
snomed_mapping AS (
    SELECT
        icd10.icd_code,
        cr.concept_id_2 AS snomed_code,
        concept.concept_name AS snomed_name
    FROM icd10
    INNER JOIN `concept_relationship` cr
        ON icd10.concept_id = cr.concept_id_1
        -- 'Maps to' is the official OMOP relationship from non-standard (ICD) to standard (SNOMED)
    INNER JOIN `concept`
        ON cr.concept_id_2 = concept.concept_id
        -- Join to get standard concept details
    WHERE cr.relationship_id = 'Maps to'
      AND cr.invalid_reason IS NULL
)
SELECT DISTINCT
    snomed_code,
    snomed_name,
    icd_code,
FROM snomed_mapping
"""
snomed_vocab = client.run(q)
snomed2icd = snomed_vocab[["snomed_code", "icd_code"]]

q = f"""
SELECT
    concept_id AS snomed_code,
    concept_name AS snomed_name
FROM concept
"""
snomed2name = client.run(q)
snomed2name = snomed2name.set_index("snomed_code")["snomed_name"]

# ### get ICD2NAME mapping

q = f"""
SELECT
    concept_id,
    LOWER(concept_code) AS icd_code,
    concept_name AS icd_name,
    concept_class_id
FROM `concept`
WHERE vocabulary_id = 'ICD10CM'
  AND LENGTH(concept_code) = 3
  AND invalid_reason IS NULL
ORDER BY concept_code;
"""
icd2name = client.run(q)
icd2name = icd2name[["icd_code", "icd_name"]]
icd2name = icd2name.set_index("icd_code")["icd_name"]

snomed2icd = snomed2icd.drop_duplicates()
snomed2icd.head()

# ### identify SNOMED codes that map to no ICD code

unmappable = (
    snomed_df.concept_id[~snomed_df.concept_id.isin(snomed2icd.snomed_code)]
    .unique()
    .tolist()
)

len(unmappable), snomed_df.concept_id.unique().shape

snomed2name

unmappable = {snomed_code: snomed2name[snomed_code] for snomed_code in unmappable}
upload_yaml(unmappable, "aou_uk/unmapped_snomed.yaml")

snomed_df = snomed_df[snomed_df.concept_id.isin(snomed2icd.snomed_code)]
snomed_df.shape

# ### identify SNOMED codes that map to more than one ICD code

# Count how many ICD codes exist for each SNOMED concept
mapping_counts = snomed2icd["snomed_code"].value_counts()
# See how many SNOMED codes have more than 1 ICD code
duplicates = mapping_counts[mapping_counts > 1]
print(f"Total unique SNOMED codes: {len(mapping_counts)}")
print(f"SNOMED codes with MULTIPLE ICD mappings: {len(duplicates)}")

# +
snomed2icd_group = snomed2icd.groupby("snomed_code")
duplicates_dict = dict()
for snomed_code in duplicates.index:
    icd_codes = snomed2icd_group.get_group(snomed_code)["icd_code"].str.lower().tolist()
    icd_names = [icd2name.get(icd_code, None) for icd_code in icd_codes]
    duplicates_dict[int(snomed_code)] = {
        "name": snomed2name[snomed_code],
        "icd_codes": dict(zip(icd_codes, icd_names)),
    }

upload_yaml(duplicates_dict, "aou_uk/many_to_one.yaml")
# -

snomed_df.concept_id.isin(duplicates.index).sum()

snomed_df = snomed_df[~snomed_df.concept_id.isin(duplicates.index)]
snomed_df.shape

# ### convert to ICD codes

snomed_df = pd.merge(
    snomed_df,
    snomed2icd.rename(columns={"snomed_code": "concept_id"}),
    on="concept_id",
    how="left",
)
snomed_df.shape

snomed_df.head()

# ## concatenate SNOMED and ICD tokens

condition_df = pd.concat((icd10_df, snomed_df), ignore_index=True)

condition_df = condition_df.sort_values(by=["person_id", "age_in_days"])
condition_df.head()

condition_df = condition_df.drop_duplicates(subset=["person_id", "icd_code"])

condition_df.shape

condition_df.person_id.unique().shape

condition_df.to_parquet(f"gs://{DATA_BUCKET}/aou_uk/condition.parquet", index=False)

# # sex

query = """
SELECT
    p.person_id,
    p.sex_at_birth_concept_id AS concept_id,
    0 AS age_in_days
FROM `person` p
WHERE p.sex_at_birth_concept_id IS NOT NULL
  AND p.sex_at_birth_concept_id NOT IN (0, 2000000009)
"""
client.dry_run(query)

sex_df = client.run(query)

sex_df.head()

sex_df.concept_id.value_counts()

sex_df["icd_code"] = sex_df.concept_id.map({45878463: "female", 45880669: "male"})

sex_df.head()

sex_df.shape

sex_df.to_parquet(f"gs://{DATA_BUCKET}/aou_uk/sex.parquet", index=False)

# # death

query = """
SELECT
    p.person_id,
    'death' AS icd_code,
    DATE_DIFF(DATE(d.death_date), DATE(p.birth_datetime), DAY) AS age_in_days
FROM `person` p
INNER JOIN `aou_death` d
    ON p.person_id = d.person_id
WHERE d.primary_death_record = TRUE  -- Ensure we don't duplicate death events
  AND d.death_date IS NOT NULL
"""
client.dry_run(query)

death_df = client.run(query)

death_df.head()

death_df.shape

death_df.to_parquet(f"gs://{DATA_BUCKET}/aou_uk/death.parquet", index=False)

# # BMI

query = """
SELECT
    m.person_id,
    DATE_DIFF(DATE(m.measurement_date), DATE(p.birth_datetime), DAY) AS age_in_days,
    CASE
        WHEN m.value_as_number > 28 THEN 'bmi_high'
        WHEN m.value_as_number > 22 THEN 'bmi_mid'
        ELSE 'bmi_low'
    END AS icd_code
FROM
    `measurement` m
INNER JOIN `person` p
    ON p.person_id = m.person_id
JOIN
    `measurement_ext` ext
    ON m.measurement_id = ext.measurement_id
WHERE
    m.measurement_concept_id = 3038553      -- The standard OMOP concept for BMI
    AND ext.src_id IN ("Staff Portal: HealthPro")
"""
client.dry_run(query)

bmi_df = client.run(query)

bmi_df.person_id.value_counts().max()

bmi_df.icd_code.value_counts()

bmi_df.to_parquet(f"gs://{DATA_BUCKET}/aou_uk/bmi.parquet", index=False)

# # combine everything

df = pd.concat((condition_df, sex_df, bmi_df, death_df), ignore_index=True)
df.shape

df.head()

df = df.sort_values(by=["person_id", "age_in_days"])

# # tokenize

import re

import yaml

with open(TOKENIZER_PATH, "r") as f:
    tokenizer = yaml.safe_load(f)
event_prefix_re = re.compile(r"^([A-Za-z]\d{2})_")
tokenizer = {
    (
        match.group(1).lower()
        if (match := event_prefix_re.match(event))
        else event.lower()
    ): code
    for event, code in tokenizer.items()
}


# ## identify tokens not found in UKB (ICD10CM tokens not in ICD10)

uniq_tokens = df.icd_code.str.lower().unique()

missing_tokens = uniq_tokens[~np.isin(uniq_tokens, list(tokenizer.keys()))]

missing_tokens = {token: icd2name[token] for token in missing_tokens}
upload_yaml(missing_tokens, "aou_uk/missing_icd_codes.yaml")

# ## apply tokenizer

df["token"] = df.icd_code.str.lower().map(tokenizer)

df = df.dropna(subset=["token"])
df.shape

df = df.drop_duplicates(subset=["person_id", "token"])
df.shape

# Diseases are the modelling endpoint; drop participants whose surviving tokens
# are only sex / bmi / death. Strictly supersedes the old single-token filter
# that previously lived in parquet_to_numpy.py.
non_disease = {"female", "male", "bmi_low", "bmi_mid", "bmi_high", "death"}
disease_pids = df.loc[
    ~df["icd_code"].str.lower().isin(non_disease), "person_id"
].unique()
n_before = df["person_id"].nunique()
df = df[df["person_id"].isin(disease_pids)]
n_after = df["person_id"].nunique()
print(f"disease-token filter: kept {n_after}/{n_before} participants")

df.head(50)

df.to_parquet(f"gs://{DATA_BUCKET}/aou_uk/data.parquet", index=False)
