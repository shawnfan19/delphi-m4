#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Build MIMIC-IV Event Sequences
#
# Extracts disease codes, procedure codes, gender, and death from
# MIMIC-IV hosp module, converts ICD-9 → ICD-10 via CMS GEMs crosswalk,
# and writes flat binary files compatible with Delphi's data format.
#
# Flags:
#   --no-procedures   Exclude ICD procedure codes
#   --no-death        Exclude death tokens

# %%
import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mimic-dir",
    default="/hps/nobackup/birney/users/sfan/delphi-data/physionet.org/files/mimiciv/3.1",
)
parser.add_argument(
    "--output-dir",
    default="/hps/nobackup/birney/users/sfan/delphi-data/mimic",
)
parser.add_argument(
    "--gems-cache",
    default="data/mimic_scripts/cache",
    help="Directory to cache the downloaded GEMs crosswalks",
)
parser.add_argument(
    "--no-procedures", action="store_true", help="Exclude procedure codes"
)
parser.add_argument("--no-death", action="store_true", help="Exclude death tokens")
args = parser.parse_args([])

MIMIC_DIR = Path(args.mimic_dir)
OUTPUT_DIR = Path(args.output_dir)
HOSP = MIMIC_DIR / "hosp"

# %% [markdown]
# ## Load MIMIC tables


# %%
def load_patients():
    df = pd.read_csv(HOSP / "patients.csv.gz")
    df["birth_year"] = df["anchor_year"] - df["anchor_age"]
    return df


def load_admissions():
    df = pd.read_csv(HOSP / "admissions.csv.gz", parse_dates=["admittime"])
    return df[["subject_id", "hadm_id", "admittime"]]


def load_diagnoses():
    return pd.read_csv(
        HOSP / "diagnoses_icd.csv.gz",
        dtype={"icd_code": str, "icd_version": int},
    )


def load_procedures():
    return pd.read_csv(
        HOSP / "procedures_icd.csv.gz",
        dtype={"icd_code": str, "icd_version": int},
        parse_dates=["chartdate"],
    )


# %%
patients = load_patients()
admissions = load_admissions()
diagnoses = load_diagnoses()

print(f"Patients: {len(patients)}")
print(f"Admissions: {len(admissions)}")
print(f"Diagnoses: {len(diagnoses)}")
print(f"  ICD-9:  {(diagnoses['icd_version'] == 9).sum()}")
print(f"  ICD-10: {(diagnoses['icd_version'] == 10).sum()}")

if not args.no_procedures:
    procedures = load_procedures()
    print(f"Procedures: {len(procedures)}")
    print(f"  ICD-9:  {(procedures['icd_version'] == 9).sum()}")
    print(f"  ICD-10: {(procedures['icd_version'] == 10).sum()}")


# %% [markdown]
# ## ICD-9 → ICD-10 conversion via GEMs crosswalk


# %%
def _download_and_parse_gems(zip_url, zip_name, csv_name, file_pattern, cache_dir):
    """Download a CMS GEMs zip, extract and parse the ICD-9→ICD-10 mapping file."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    gems_csv = cache_dir / csv_name
    if gems_csv.exists():
        return pd.read_csv(gems_csv, dtype=str)

    zip_path = cache_dir / zip_name
    if not zip_path.exists():
        print(f"Downloading GEMs crosswalk from {zip_url}...")
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Done.")

    with zipfile.ZipFile(zip_path) as zf:
        gem_files = [n for n in zf.namelist() if file_pattern in n.lower()]
        assert (
            gem_files
        ), f"No {file_pattern} file found in zip. Contents: {zf.namelist()}"
        with zf.open(gem_files[0]) as f:
            raw = f.read().decode("utf-8").strip().split("\n")

    records = []
    for line in raw:
        parts = line.split()
        if len(parts) >= 2:
            records.append({"icd9": parts[0].strip(), "icd10": parts[1].strip()})

    gems = pd.DataFrame(records)
    gems.to_csv(gems_csv, index=False)
    print(f"Parsed {len(gems)} GEMs mappings → {gems_csv}")
    return gems


def download_diagnosis_gems(cache_dir):
    """Download CMS 2018 ICD-9-CM → ICD-10-CM (diagnosis) GEMs."""
    return _download_and_parse_gems(
        zip_url="https://www.cms.gov/Medicare/Coding/ICD10/Downloads/2018-ICD-10-CM-General-Equivalence-Mappings.zip",
        zip_name="gems_dx_2018.zip",
        csv_name="gems_dx_i9to10.csv",
        file_pattern="i9gem",
        cache_dir=cache_dir,
    )


def download_procedure_gems(cache_dir):
    """Download CMS 2018 ICD-9-PCS → ICD-10-PCS (procedure) GEMs."""
    return _download_and_parse_gems(
        zip_url="https://www.cms.gov/Medicare/Coding/ICD10/Downloads/2018-ICD-10-PCS-General-Equivalence-Mappings.zip",
        zip_name="gems_px_2018.zip",
        csv_name="gems_px_i9to10.csv",
        file_pattern="gem_i9pcs",
        cache_dir=cache_dir,
    )


# %%
dx_gems = download_diagnosis_gems(args.gems_cache)
print(f"Diagnosis GEMs: {len(dx_gems)} mappings")

if not args.no_procedures:
    px_gems = download_procedure_gems(args.gems_cache)
    print(f"Procedure GEMs: {len(px_gems)} mappings")


# %% [markdown]
# ## Convert ICD codes to 3-char tokens


# %%
def convert_icd_codes(df, gems, prefix=""):
    """Convert ICD-9/ICD-10 codes to lowercase 3-char tokens.

    ICD-10 codes are truncated directly.
    ICD-9 codes are mapped via GEMs (full code → full ICD-10 → truncate).
    An optional prefix (e.g. 'px_') is prepended to all tokens.
    """
    df = df.copy()

    # --- ICD-10: truncate to 3 chars ---
    icd10_mask = df["icd_version"] == 10
    df.loc[icd10_mask, "token"] = df.loc[icd10_mask, "icd_code"].str[:3].str.lower()

    # --- ICD-9: look up full code in GEMs, then truncate result ---
    icd9_mask = df["icd_version"] == 9
    gems_dedup = gems.drop_duplicates(subset="icd9", keep="first")
    gems_map = gems_dedup.set_index("icd9")["icd10"]

    icd9_codes = df.loc[icd9_mask, "icd_code"].str.strip()
    icd10_mapped = icd9_codes.map(gems_map)
    df.loc[icd9_mask, "token"] = icd10_mapped.str[:3].str.lower()

    n_total_icd9 = icd9_mask.sum()
    n_unmapped = df.loc[icd9_mask, "token"].isna().sum()
    label = prefix.rstrip("_").upper() or "Diagnosis"
    print(f"{label} ICD-9 → ICD-10 conversion:")
    print(f"  Mapped:   {n_total_icd9 - n_unmapped}/{n_total_icd9}")
    print(
        f"  Unmapped: {n_unmapped}/{n_total_icd9} ({n_unmapped / n_total_icd9 * 100:.1f}%)"
    )

    df = df.dropna(subset=["token"])

    if prefix:
        df["token"] = prefix + df["token"]

    return df


# %%
diagnoses = convert_icd_codes(diagnoses, dx_gems, prefix="")

if not args.no_procedures:
    procedures = convert_icd_codes(procedures, px_gems, prefix="px_")


# %% [markdown]
# ## Compute age in days


# %%
def age_in_days(date_series, birth_year_series):
    """Compute age in days from a date and a birth year (approx Jan 1)."""
    birth_date = pd.to_datetime(
        birth_year_series.astype(str) + "-01-01", format="%Y-%m-%d"
    )
    return (date_series - birth_date).dt.days


# %% [markdown]
# ## Build event sequences


# %%
def build_diagnosis_events(diagnoses, patients, admissions):
    """Build diagnosis events: join with admissions for timestamps."""
    events = diagnoses.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    events = events.merge(
        patients[["subject_id", "birth_year"]], on="subject_id", how="inner"
    )
    events["age_days"] = age_in_days(events["admittime"], events["birth_year"])
    events = events[events["age_days"] >= 0]
    return events[["subject_id", "token", "age_days"]]


def build_procedure_events(procedures, patients):
    """Build procedure events: use chartdate directly (no admission join needed)."""
    events = procedures.merge(
        patients[["subject_id", "birth_year"]], on="subject_id", how="inner"
    )
    events["age_days"] = age_in_days(events["chartdate"], events["birth_year"])
    events = events[events["age_days"] >= 0]
    return events[["subject_id", "token", "age_days"]]


def build_death_events(patients):
    """Build death events from patients.dod."""
    dead = patients[patients["dod"].notna()].copy()
    dead["dod"] = pd.to_datetime(dead["dod"])
    dead["age_days"] = age_in_days(dead["dod"], dead["birth_year"])
    dead = dead[dead["age_days"] >= 0]
    dead["token"] = "death"
    return dead[["subject_id", "token", "age_days"]]


def build_gender_events(patients, subject_ids):
    """Build gender tokens at age 0 for the given subject_ids."""
    gender_map = {"F": "female", "M": "male"}
    df = patients[patients["subject_id"].isin(subject_ids)].copy()
    df["token"] = df["gender"].map(gender_map)
    df["age_days"] = 0
    return df[["subject_id", "token", "age_days"]]


# %%
# Collect all event sources
event_parts = [build_diagnosis_events(diagnoses, patients, admissions)]

if not args.no_procedures:
    event_parts.append(build_procedure_events(procedures, patients))

# Combine clinical events, deduplicate, then add gender + death
events = pd.concat(event_parts, ignore_index=True)
events = events.sort_values(["subject_id", "age_days"])
n_before = len(events)
events = events.drop_duplicates(subset=["subject_id", "token"], keep="first")
print(f"Deduplication: {n_before} → {len(events)} events")

# Determine which subjects have at least one clinical event
subjects_with_events = set(events["subject_id"].unique())

# Add gender
events = pd.concat(
    [events, build_gender_events(patients, subjects_with_events)], ignore_index=True
)

# Add death (only for patients already in the dataset)
if not args.no_death:
    death_events = build_death_events(patients)
    death_events = death_events[death_events["subject_id"].isin(subjects_with_events)]
    events = pd.concat([events, death_events], ignore_index=True)
    print(f"Death events: {len(death_events)}")

events = events.sort_values(["subject_id", "age_days"]).reset_index(drop=True)

print(f"\nTotal events: {len(events)}")
print(f"Unique patients: {events['subject_id'].nunique()}")
print(f"Unique tokens: {events['token'].nunique()}")


# %% [markdown]
# ## Build tokenizer with descriptions


# %%
def _clean_description(s):
    return (
        s.str.lower()
        .str.replace(r"[^a-z0-9 ,\-]", "", regex=True)
        .str.replace(" ", "_")
    )


def build_tokenizer(events):
    """Build tokenizer: token_name → int ID.

    Diagnosis tokens: 'a00_(cholera)' style
    Procedure tokens: 'px_5a1_(extracorporeal_or_systemic_assistance_and_performance,_performance)' style
    Special tokens: padding, no_event, female, male, death
    """
    # --- Load ICD-10-CM descriptions (diagnoses) ---
    d_dx = pd.read_csv(HOSP / "d_icd_diagnoses.csv.gz", dtype={"icd_code": str})
    dx_desc = d_dx[d_dx["icd_version"] == 10].copy()
    dx_desc["code_3"] = dx_desc["icd_code"].str[:3].str.lower()
    dx_desc_map = (
        dx_desc.sort_values("icd_code")
        .drop_duplicates(subset="code_3", keep="first")
        .set_index("code_3")["long_title"]
        .pipe(_clean_description)
    )

    # --- Load ICD-10-PCS descriptions (procedures) ---
    d_px = pd.read_csv(HOSP / "d_icd_procedures.csv.gz", dtype={"icd_code": str})
    px_desc = d_px[d_px["icd_version"] == 10].copy()
    px_desc["code_3"] = px_desc["icd_code"].str[:3].str.lower()
    px_desc_map = (
        px_desc.sort_values("icd_code")
        .drop_duplicates(subset="code_3", keep="first")
        .set_index("code_3")["long_title"]
        .pipe(_clean_description)
    )

    # --- Special tokens ---
    special = ["female", "male", "death"]
    tokenizer = {"padding": 0, "no_event": 1}
    code2id = {}
    for tok in special:
        if tok in events["token"].values:
            tokenizer[tok] = len(tokenizer)
            code2id[tok] = tokenizer[tok]

    # --- Disease tokens (no prefix) ---
    dx_tokens = sorted(
        t
        for t in events["token"].unique()
        if t not in special and not t.startswith("px_")
    )
    for code in dx_tokens:
        desc = dx_desc_map.get(code, "")
        name = f"{code}_({desc})" if desc else code
        tokenizer[name] = len(tokenizer)
        code2id[code] = tokenizer[name]

    # --- Procedure tokens (px_ prefix) ---
    px_tokens = sorted(t for t in events["token"].unique() if t.startswith("px_"))
    for token in px_tokens:
        code_3 = token[3:]  # strip px_ prefix to look up description
        desc = px_desc_map.get(code_3, "")
        name = f"{token}_({desc})" if desc else token
        tokenizer[name] = len(tokenizer)
        code2id[token] = tokenizer[name]

    return tokenizer, code2id


# %%
tokenizer, code2id = build_tokenizer(events)
print(f"Vocabulary size: {len(tokenizer)}")


# %% [markdown]
# ## Write output files


# %%
def write_output(events, tokenizer, code2id, output_dir):
    """Write data.bin, time.bin, p2i.csv, tokenizer.yaml, participant splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode tokens
    events = events.copy()
    events["token_id"] = events["token"].map(code2id)
    unmapped = events["token_id"].isna().sum()
    if unmapped > 0:
        missing = events[events["token_id"].isna()]["token"].unique()
        print(f"WARNING: {unmapped} events with unmapped tokens: {missing[:10]}")
        events = events.dropna(subset=["token_id"])
    events["token_id"] = events["token_id"].astype(np.uint32)
    events["age_days"] = events["age_days"].astype(np.uint32)

    # Sort and build per-patient arrays
    events = events.sort_values(["subject_id", "age_days"]).reset_index(drop=True)

    all_tokens = events["token_id"].values
    all_times = events["age_days"].values

    # Build p2i index
    grouped = events.groupby("subject_id", sort=True)
    pids = []
    start_positions = []
    seq_lens = []
    pos = 0
    for subject_id, group in grouped:
        pids.append(subject_id)
        n = len(group)
        start_positions.append(pos)
        seq_lens.append(n)
        pos += n

    pids = np.array(pids, dtype=np.uint32)

    # Write binary files
    all_tokens.tofile(output_dir / "data.bin")
    all_times.tofile(output_dir / "time.bin")

    p2i = pd.DataFrame(
        {
            "pid": pids,
            "start_pos": start_positions,
            "seq_len": seq_lens,
        }
    ).set_index("pid")
    p2i.to_csv(output_dir / "p2i.csv")

    # Write tokenizer
    with open(output_dir / "tokenizer.yaml", "w") as f:
        yaml.dump(tokenizer, f, default_flow_style=False, sort_keys=False)

    # Write participant splits (80/20)
    participants_dir = output_dir / "participants"
    participants_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(pids))
    train_size = int(0.8 * len(pids))
    train_pids = np.sort(pids[perm[:train_size]])
    val_pids = np.sort(pids[perm[train_size:]])

    pids.tofile(participants_dir / "all.bin")
    train_pids.tofile(participants_dir / "train_fold.bin")
    val_pids.tofile(participants_dir / "val_fold.bin")

    print(f"\nOutput written to {output_dir}")
    print(f"  data.bin:  {len(all_tokens)} tokens")
    print(f"  time.bin:  {len(all_times)} timestamps")
    print(f"  p2i.csv:   {len(pids)} patients")
    print(f"  tokenizer: {len(tokenizer)} entries")
    print(f"  train/val: {len(train_pids)}/{len(val_pids)}")


# %%
write_output(events, tokenizer, code2id, OUTPUT_DIR)

# %%
