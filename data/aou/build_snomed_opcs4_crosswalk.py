"""Extract a slim SNOMED -> OPCS-4 crosswalk from an OHDSI Athena vocabulary bundle.

OPCS-4 is not loaded in the AoU CDR, so `data/aou/operations.py` needs an external
map to push AoU's SNOMED procedures into the UKB `summary_ops` (OPCS-4) token space.
Download an Athena bundle with **OPCS4 + SNOMED** selected, unzip it into the bucket
(e.g. gs://{DATA_BUCKET}/vocab/athena/), then run on the workbench:

    python build_snomed_opcs4_crosswalk.py vocab/athena

Both paths are relative to DATA_BUCKET: it reads the Athena CSVs from
gs://{DATA_BUCKET}/<athena_dir>/ and writes the crosswalk to
gs://{DATA_BUCKET}/aou_uk/snomed_to_opcs4.csv (override with -o).

NOTE: OPCS-4 is UK Crown copyright — it lives in the bucket, not in git.
"""

import argparse

import pandas as pd

# CONCEPT_RELATIONSHIP is huge once SNOMED is in the bundle; filter it in chunks.
_CHUNKSIZE = 2_000_000


def build_crosswalk(athena_base: str) -> pd.DataFrame:
    # Athena ships OMOP vocab files as tab-delimited (despite the .csv extension).
    # `athena_base` is a directory prefix (local or gs://) holding the bundle CSVs.
    concept = pd.read_csv(
        f"{athena_base}/CONCEPT.csv",
        sep="\t",
        dtype=str,
        usecols=["concept_id", "concept_code", "vocabulary_id"],
    )
    opcs = concept.loc[
        concept["vocabulary_id"] == "OPCS4", ["concept_id", "concept_code"]
    ]
    snomed_ids = concept.loc[
        concept["vocabulary_id"] == "SNOMED", ["concept_id"]
    ].rename(columns={"concept_id": "snomed_id"})

    # OPCS-4 (concept_id_1) 'Maps to' standard SNOMED (concept_id_2); keep valid rows only.
    maps = []
    for chunk in pd.read_csv(
        f"{athena_base}/CONCEPT_RELATIONSHIP.csv",
        sep="\t",
        dtype=str,
        usecols=["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"],
        chunksize=_CHUNKSIZE,
    ):
        maps.append(
            chunk[
                (chunk["relationship_id"] == "Maps to")
                & (chunk["invalid_reason"].isna())
            ]
        )
    maps = pd.concat(maps, ignore_index=True)

    xwalk = maps.merge(opcs, left_on="concept_id_1", right_on="concept_id").merge(
        snomed_ids, left_on="concept_id_2", right_on="snomed_id"
    )
    xwalk = (
        xwalk[["concept_id_2", "concept_code"]]
        .rename(
            columns={"concept_id_2": "snomed_concept_id", "concept_code": "opcs4_code"}
        )
        .astype({"snomed_concept_id": "int64"})
        .drop_duplicates()
        .sort_values(["snomed_concept_id", "opcs4_code"])
        .reset_index(drop=True)
    )
    return xwalk


def main():
    from utils import DATA_BUCKET

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "athena_dir",
        help="unzipped Athena bundle dir, relative to DATA_BUCKET "
        "(read from gs://<DATA_BUCKET>/<athena_dir>/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="aou_uk/snomed_to_opcs4.csv",
        help="output path relative to DATA_BUCKET; written to "
        "gs://<DATA_BUCKET>/<output> (default: aou_uk/snomed_to_opcs4.csv)",
    )
    args = parser.parse_args()

    xwalk = build_crosswalk(f"gs://{DATA_BUCKET}/{args.athena_dir}")
    print(
        f"{len(xwalk)} SNOMED->OPCS4 rows; "
        f"{xwalk['snomed_concept_id'].nunique()} distinct SNOMED, "
        f"{xwalk['opcs4_code'].nunique()} distinct OPCS-4 codes"
    )
    dest = f"gs://{DATA_BUCKET}/{args.output}"
    xwalk.to_csv(dest, index=False)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
