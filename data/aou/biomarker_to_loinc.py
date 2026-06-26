"""Map each biomarker in data/biomarker.yaml to its LOINC code via the OMOP
`concept` table. Run on the AoU workbench. Writes data/biomarker_loinc.yaml
(name -> LOINC code, or list of codes when a biomarker has several concept ids).

LOINC is the OMOP standard vocabulary for the Measurement domain, so a
measurement concept's `concept_code` *is* its LOINC code whenever
`vocabulary_id == 'LOINC'`. Concept ids that are not LOINC (or absent from the
`concept` table) have no LOINC code and are only reported, not written.
"""

import os
from pathlib import Path

import yaml
from utils import WORKSPACE_CDR, Client

CWD = Path(os.getcwd())
client = Client(dataset=WORKSPACE_CDR)

with open(CWD.parent / "biomarker.yaml") as f:
    biomarkers = yaml.safe_load(f)

# name -> list of OMOP concept ids (skip entries without an AoU mapping)
name2ids = {name: v["aou"]["id"] for name, v in biomarkers.items() if "aou" in v}
all_ids = sorted({i for ids in name2ids.values() for i in ids})

ids_str = ", ".join(str(i) for i in all_ids)
df = client.run(
    f"""
    SELECT concept_id, concept_code, vocabulary_id, concept_name
    FROM `concept`
    WHERE concept_id IN ({ids_str})
    """
)

# OMOP concept id -> LOINC code (only for genuine LOINC concepts)
loinc = {
    int(r.concept_id): r.concept_code
    for r in df.itertuples()
    if r.vocabulary_id == "LOINC"
}

mapping, problems = {}, {}
for name, ids in name2ids.items():
    codes = [loinc[i] for i in ids if i in loinc]
    missing = [i for i in ids if i not in loinc]
    if missing:
        problems[name] = missing  # non-LOINC or not found in `concept`
    if codes:
        mapping[name] = codes[0] if len(codes) == 1 else codes

out = CWD.parent / "biomarker_loinc.yaml"
with open(out, "w") as f:
    yaml.dump(mapping, f, sort_keys=True, default_flow_style=False)
print(f"wrote {len(mapping)} biomarkers -> {out}")

if problems:
    print("\nconcept ids with no LOINC code (non-LOINC vocab or not in `concept`):")
    for name, ids in problems.items():
        for i in ids:
            row = df[df.concept_id == i]
            why = f"vocab={row.vocabulary_id.iloc[0]}" if len(row) else "not found"
            print(f"  {name}: {i} ({why})")
