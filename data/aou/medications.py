# +
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from utils import DATA_BUCKET, WORKSPACE_CDR, Client, upload_yaml
# -

client = Client(dataset=WORKSPACE_CDR)

client.list_columns("drug_exposure")

client.list_rows("drug_exposure")

client.list_columns("concept_ancestor")

client.list_rows("concept_ancestor")

client.run("""
  SELECT concept_class_id, COUNT(*) AS n
  FROM `concept`
  WHERE vocabulary_id = 'ATC'
  GROUP BY concept_class_id
  ORDER BY concept_class_id
  """)

query = """
SELECT
  de.person_id,
  atc.concept_code AS atc_code,          -- 5-char ATC 4th code, e.g. 'A10BA'
  atc.concept_name AS atc_name,
  MIN(DATE_DIFF(DATE(de.drug_exposure_start_date),
                DATE(p.birth_datetime), DAY)) AS age_in_days
FROM `drug_exposure` de
INNER JOIN `person` p
  ON p.person_id = de.person_id
-- climb the OMOP hierarchy from the standard RxNorm/RxNorm-Ext drug to its ATC ancestors
INNER JOIN `concept_ancestor` ca
  ON ca.descendant_concept_id = de.drug_concept_id
INNER JOIN `concept` atc
  ON atc.concept_id = ca.ancestor_concept_id
WHERE de.drug_concept_id != 0                  -- drop unmapped drugs
AND de.drug_exposure_start_date IS NOT NULL
AND atc.vocabulary_id = 'ATC'
AND atc.concept_class_id = 'ATC 4th'         -- the level UKB tokenizes at
-- GROUP BY collapses the many-to-many fan-out to one row per (person, ATC-4th),
-- and MIN(age) takes the earliest occurrence (mirrors keep="first" in gather_prescriptions)
GROUP BY de.person_id, atc_code, atc_name
ORDER BY de.person_id, age_in_days
"""
client.dry_run(query)

df = client.run(query)

df.shape

df.columns

out = f"gs://{DATA_BUCKET}/aou"
df.to_parquet(f"{out}/atc.parquet", index=False)

# +
try:
    _CORE_DIR = Path(__file__).resolve().parent
except NameError:  # Jupyter cell execution
    _CORE_DIR = Path(os.getcwd())
TOKENIZER_PATH = _CORE_DIR.parent / "ukb" / "dictionary" / "prescriptions_tokenizer.yaml"

with open(TOKENIZER_PATH) as f:
  tokenizer = yaml.safe_load(f)

# +
# OMOP ATC codes are upper-case; normalise both sides defensively
tokenizer = {k.upper(): v for k, v in tokenizer.items()}
df["atc_code"] = df["atc_code"].str.upper()

# log ATC-4th codes seen in AoU but absent from the UKB token space (cf. missing_icd_codes.yaml)
seen = df[["atc_code", "atc_name"]].drop_duplicates()
missing = dict(seen.loc[~seen["atc_code"].isin(tokenizer)].itertuples(index=False, name=None))
# -

upload_yaml(missing, "aou_uk/expansion_packs/prescriptions/missing_atc_codes.yaml")

# apply tokenizer, drop AoU-only codes (keep the shared UKB vocab), enforce age sanity
df["token"] = df["atc_code"].map(tokenizer)
df = df.dropna(subset=["token"])
df["token"] = df["token"].astype(int)
df = df[df["age_in_days"] >= 0]
df = df.drop_duplicates(subset=["person_id", "token"])
df = df.sort_values(["person_id", "age_in_days"])

out = f"gs://{DATA_BUCKET}/aou_uk/expansion_packs/prescriptions"
df[["person_id", "age_in_days", "token"]].to_parquet(f"{out}/data.parquet", index=False)
upload_yaml(tokenizer, "aou_uk/expansion_packs/prescriptions/tokenizer.yaml")  # AOUExpansionPack reads this




