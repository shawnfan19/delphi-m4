# +
import os
import re
from pathlib import Path

import pandas as pd
import yaml
from utils import DATA_BUCKET, WORKSPACE_CDR, Client, upload_yaml

# -

client = Client(dataset=WORKSPACE_CDR)

client.list_columns("procedure_occurrence")

client.list_rows("procedure_occurrence")

# # procedure tokens (OPCS-4 3-char, shared token space with UKB `ops`)

# AoU procedures are standardized to SNOMED; first occurrence per (person, SNOMED).
query = """
SELECT
  po.person_id,
  po.procedure_concept_id AS snomed_concept_id,
  MIN(DATE_DIFF(DATE(po.procedure_date), DATE(p.birth_datetime), DAY)) AS age_in_days
FROM `procedure_occurrence` po
INNER JOIN `person` p
  ON p.person_id = po.person_id
WHERE po.procedure_concept_id != 0          -- drop unmapped procedures
  AND po.procedure_date IS NOT NULL
GROUP BY po.person_id, snomed_concept_id     -- MIN(age) => earliest occurrence
ORDER BY po.person_id, age_in_days
"""
client.dry_run(query)

df = client.run(query)
df.shape

# # map SNOMED -> OPCS-4 via the external Athena crosswalk
# (OPCS4 is not in the AoU CDR, so we can't traverse concept_relationship in-warehouse.)

xwalk = pd.read_csv(
    f"gs://{DATA_BUCKET}/aou_uk/snomed_to_opcs4.csv",
    dtype={"snomed_concept_id": "int64", "opcs4_code": str},
)

df = df.merge(xwalk, on="snomed_concept_id", how="inner")  # many-to-many fan-out
df.shape

# truncate to the OPCS-4 3-char category, matching gather_summary_operations.py (str[0:3])
df["opcs3"] = df["opcs4_code"].str[:3].str.lower()

out = f"gs://{DATA_BUCKET}/aou_uk/expansion_packs/ops"
df.to_parquet(f"{out}/raw.parquet", index=False)

# # tokenize against the UKB `ops` vocabulary (shared token space)

try:
    _CORE_DIR = Path(__file__).resolve().parent
except NameError:  # Jupyter cell execution
    _CORE_DIR = Path(os.getcwd())
TOKENIZER_PATH = _CORE_DIR.parent / "ukb" / "dictionary" / "ops_tokenizer.yaml"

with open(TOKENIZER_PATH) as f:
    tokenizer = yaml.safe_load(f)

# UKB tokenizer keys are `{opcs3char}_{name}`; derive a {3-char code -> token} lookup
# from the leading prefix (same idiom as core.py's event-prefix regex).
prefix_re = re.compile(r"^([a-z]\d{2})_")
code2token = {
    m.group(1): tok
    for key, tok in tokenizer.items()
    if (m := prefix_re.match(key.lower()))
}

# log OPCS-4 categories present in AoU but absent from the UKB vocab
missing = sorted(set(df["opcs3"]) - set(code2token))
upload_yaml(missing, "aou_uk/expansion_packs/ops/missing_opcs4_codes.yaml")

n_mapped = len(df)
df["token"] = df["opcs3"].map(code2token)
df = df.dropna(subset=["token"])
df["token"] = df["token"].astype(int)
print(
    f"tokenized {len(df)}/{n_mapped} OPCS-mapped rows; "
    f"{len(missing)} OPCS-4 categories missing from the UKB vocab"
)

# first occurrence per (person, token): collapse the SNOMED -> multi-OPCS fan-out
df = (
    df.groupby(["person_id", "token"], as_index=False)["age_in_days"]
    .min()
    .sort_values(["person_id", "age_in_days"])
)
df.shape

df[["person_id", "age_in_days", "token"]].to_parquet(f"{out}/data.parquet", index=False)
upload_yaml(
    tokenizer, "aou_uk/expansion_packs/ops/tokenizer.yaml"
)  # AOUExpansionPack reads this
