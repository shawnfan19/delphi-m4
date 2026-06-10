# UKB Data Extraction Pipeline

This document describes how raw UK Biobank phenotype data is extracted and converted into Delphi's flat binary format. For documentation on the dataset classes that consume these files, see `AGENTS/DATA.md`.

---

## Overview

```
RAW UKB DATA (DNAnexus RAP / EMBL-EBI Codon)
    │
    ├─ gather_core.py ──────────► core events: data.bin, time.bin, p2i.csv, tokenizer.yaml, participant splits
    │
    ├─ gather.py ───────────────► biomarkers/<modality>/{data.bin, p2i.csv, features.yaml}
    │
    ├─ gather_family_history.py ─┐
    ├─ gather_female_specific_factors.py
    ├─ gather_medications.py     ├► expansion_packs/<name>/{data.bin, time.bin, p2i.csv, tokenizer.yaml}
    ├─ gather_operations.py      │
    ├─ gather_summary_operations.py
    └─ gather_prescriptions.py ──┘
```

Three output types:
1. **Core events** — discrete disease/demographic tokens (ICD-10, sex, lifestyle, cancer, death)
2. **Biomarkers** — continuous feature vectors per modality (blood panels, diet, PRS, etc.)
3. **Expansion packs** — additional discrete tokens (operations, medications, family history, etc.)

---

## Platform Detection

`delphi/env.py` detects the execution environment:

| Environment | Detection | `DELPHI_DATA_DIR` |
|-------------|-----------|-------------------|
| DNAnexus RAP | `DX_PROJECT_CONTEXT_ID` env var present | `/opt/data` |
| EMBL-EBI Codon | otherwise | `$DELPHI_DATA_DIR` env var |

`data/ukb_scripts/utils.py` conditionally imports platform-specific implementations:

| Module | Platform | Data access |
|--------|----------|-------------|
| `utils_rap.py` | RAP | PySpark + `dxdata` library |
| `utils_codon.py` | Codon | Tab-separated files in `$DELPHI_DATA_DIR/ukb/tab/` |

Both expose the same interface: `load_fid()`, `load_fids()`, `load_coding()`, `month_of_birth()`, `assessment_age()`, and the visit list:

```python
VISITS = ["birth", "init_assess", "1st_repeat_assess", "img", "1st_repeat_img"]
```

Assessment ages are computed as `(assessment_date - month_of_birth).days` per visit, using UKB field 53 (date of attending assessment centre).

---

## Core Event Extraction (`gather_core.py`)

Runs on the **RAP only** (uses PySpark and `dxdata`). Produces the base discrete event sequences.

### Data extracted

| Category | UKB Fields | Tokens |
|----------|-----------|--------|
| Sex | Field: Sex | `female`, `male` |
| BMI | Field: `p21001_i0` | `bmi_low` (<22), `bmi_mid` (22–28), `bmi_high` (>28) |
| Smoking | Field: `p1239_i0` | `smoking_low` (other), `smoking_mid` (=2), `smoking_high` (=1) |
| Alcohol | Field: `p1558_i0` | `alcohol_low` (≥4), `alcohol_mid` (<4), `alcohol_high` (=1) |
| ICD-10 diagnoses | First-occurrence fields 130000–132604 | 3-character ICD-10 codes (e.g. `e11`) |
| Cancer | Fields 40006 (type) / 40005 (date), 22 instances | 3-character cancer codes (truncated from 4) |
| Death | Date of death (death table) | `death` token |

### Processing steps

1. Connect to UKB dataset via `dxdata`, extract all fields in one PySpark query
2. Discretize lifestyle factors using thresholds above
3. Truncate cancer codes to 3 characters (`C501` → `C50`)
4. Stack all events into long format: `(eid, token, date)` using PySpark `stack()`
5. Convert dates to **age in days**: `datediff(event_date, date_of_birth)`
6. Exclude sentinel dates (`1900-01-01`, `1901-01-01`, `1902-02-02`, `1903-03-03`, `1909-09-09`, `2037-07-07`) and negative ages
7. Deduplicate: one occurrence per (eid, token)
8. Map token strings to integer IDs via `dictionary/tokenizer.yaml`
9. Sort by (eid, age), write `data.bin` (uint32), `time.bin` (uint32), `p2i.csv`
10. Split participants 80/20 train/val (seed=42), write to `participants/`
11. Upload to workspace via `dx upload`

---

## Biomarker Extraction (`gather.py`)

Extracts continuous-valued biomarker panels. Runs on either platform.

### Configuration

`dictionary/panel.yaml` defines each modality:

```yaml
wbc:                           # modality name (matches Modality enum)
  fids:
    30160: basophill_count     # UKB field ID: feature name
    30220: basophill_percentage
    ...                        # 31 features total
  visits:
    - init_assess
    - 1st_repeat_assess
    - img
```

### Modalities defined in `panel.yaml`

| Modality | Features | Visits |
|----------|----------|--------|
| `wbc` | 31 (full blood count) | init_assess, 1st_repeat_assess, img |
| `lipid` | 4 (cholesterol, HDL, LDL, triglycerides) | init_assess, 1st_repeat_assess |
| `lft` | 7 (liver function) | init_assess, 1st_repeat_assess |
| `renal` | 6 (renal function) | init_assess, 1st_repeat_assess |
| `hba1c` | 1 | init_assess, 1st_repeat_assess |
| `crp` | 1 | init_assess, 1st_repeat_assess |
| `urate` | 1 | init_assess, 1st_repeat_assess |
| `cysc` | 1 (cystatin C) | init_assess, 1st_repeat_assess |
| `apo` | 2 (apolipoprotein A, B) | init_assess, 1st_repeat_assess |
| `vitd` | 1 (vitamin D) | init_assess, 1st_repeat_assess |
| `dht` | 1 (testosterone) | init_assess, 1st_repeat_assess |
| `shbg` | 1 | init_assess, 1st_repeat_assess |
| `igf1` | 1 | init_assess, 1st_repeat_assess |
| `nak` | 2 (sodium, potassium) | init_assess, 1st_repeat_assess |
| `creat` | 1 (creatinine, urine) | init_assess, 1st_repeat_assess |
| `albu` | 1 (albumin, urine) | init_assess, 1st_repeat_assess |
| `diet` | 16 (food frequency) | init_assess, 1st_repeat_assess, img, 1st_repeat_img |
| `prs` | 35 (polygenic risk scores) | birth |
| `met` | 3 (walking, moderate, vigorous activity) | init_assess, 1st_repeat_assess, img, 1st_repeat_img |
| `telomere` | 1 | init_assess, 1st_repeat_assess, img |

Abdominal fat modalities (`abdo_fat_cross`, `abdo_fat_long`) are defined in the `Modality` enum but commented out in `panel.yaml`.

### Processing steps

1. Preload all field IDs in bulk via `load_fids()`
2. For each modality, `load_biomarker_df()` loads field data and reshapes to long format indexed by `(pid, visit)` using `index_by_visit()`
3. Diet-specific handling: `-3` (prefer not answer) → NaN, `-1` (do not know) → NaN, `-10` (less than one) → `0`
4. `build_biomarker()` writes the output:
   - Filters out participants not in the Delphi cohort (`all_ukb_participants()`)
   - Removes rows with NaN in timestamps or feature values
   - Timestamps are assessment ages (days) from `_long_assessment_age()`
   - Writes `data.bin` (float32, flat), `p2i.csv` (pid, visit, start_pos, seq_len, time), `features.yaml`

---

## Expansion Pack Scripts

Each script follows the same pattern: extract UKB fields → map to local token vocabulary → call `build_expansion_pack()`.

`build_expansion_pack()` (in `utils.py`) writes:
- `data.bin` (uint32, local token IDs)
- `time.bin` (uint32, timestamps in days)
- `p2i.csv` (pid → start_pos, seq_len; initialized from `all_ukb_participants()` so all participants have an entry)
- `tokenizer.yaml` (token name → local integer ID)

### `gather_family_history.py` → `expansion_packs/family_hx/`

- **UKB fields**: 20107 (father), 20110 (mother), 20111 (sibling illness history)
- **Coding**: UKB coding scheme 1010, mapped via `dictionary/family_hx_coding.yaml` (13 conditions: heart disease, stroke, lung cancer, bowel cancer, breast cancer, chronic bronchitis/emphysema, alzheimer's/dementia, parkinson's, diabetes, high blood pressure, severe depression, prostate cancer, hip fracture)
- **Timestamps**: all set to 0 (birth) — family history is inherent information
- **Special handling**: negative coding values and NaN excluded

### `gather_female_specific_factors.py` → `expansion_packs/fsf/`

- **Participants**: females only (sex field 31 = 0)
- **Tokens**: `menarche`, `menopause`, `parity_zero`/`one`/`two`/`three_or_more`, `first_live_birth`, `miscarriage_one`/`miscarriage_multiple`
- **UKB fields**: 2714 (menarche age), 2724 (had menopause), 3581 (menopause age), 2734 (live births count), 3872/2754 (first birth age), 2774 (failed pregnancy), 3839 (miscarriage count)
- **Timestamps**: menarche/menopause → actual age in days (`age * DAYS_PER_YEAR`); parity/miscarriage → assessment age
- **Special handling**: coded missing values (-3, -1, -4) skipped; unknown/ambiguous menopause status skipped

### `gather_medications.py` → `expansion_packs/prescriptions_hx/`

- **UKB field**: 20003 (self-reported medications, coding scheme 4)
- **Vocabulary filtering**: only medications with >100 total occurrences; coding 99999 excluded
- **Token naming**: `{coding}_{meaning}` (e.g. `1140879616_aspirin`)
- **Timestamps**: assessment age (all medications from same visit share the same timestamp)
- **Uses polars** for the token mapping step

### `gather_operations.py` → `expansion_packs/self_report_ops/`

- **UKB fields**: 20004 (self-reported operations, coding scheme 5), 20011 (age at operation)
- **Timestamps**: reported age × `DAYS_PER_YEAR` (converted from years to days)
- **Filtering**: coding values ≤0 or 99999 or time ≤0 excluded

### `gather_summary_operations.py` → `expansion_packs/ops/`

- **UKB fields**: 41200 (hospital OPCS-4 operation codes, coding scheme 240), 41260 (operation dates)
- **Grouping**: child OPCS-4 codes grouped to parent-level (3-character codes); "Chapter" entries excluded
- **Timestamps**: `(operation_date - month_of_birth).days`
- **Uses polars** for mapping and date arithmetic

### `gather_prescriptions.py` → `expansion_packs/prescriptions/`

- **Source**: GP prescription data from `gp_scripts.txt` (primary care linkage)
- **Token mapping**: BNF presentation codes → BNF subparagraph-level tokens; falls back to Read v2 codes when BNF code is missing
- **Processing**: reads in 1M-row chunks with last-participant hold-out to avoid splitting a participant across chunks
- **Deduplication**: first occurrence only per (eid, bnf_code)
- **Timestamps**: `(issue_date - month_of_birth).days`
- **Note**: reads from `$DELPHI_DATA_DIR/primary_care/` rather than the standard tab directory

---

## Dictionary Files (`data/ukb_scripts/dictionary/`)

### `tokenizer.yaml`

Base vocabulary (~1270 tokens). Token 0 = padding, token 1 = no_event (synthetic). Remaining tokens:

| Range | Content |
|-------|---------|
| 2–3 | `female`, `male` |
| 4–12 | Lifestyle: `bmi_low/mid/high`, `smoking_low/mid/high`, `alcohol_low/mid/high` |
| 13+ | ICD-10 3-character disease codes (e.g. `a00_(cholera): 13`) |

Sex and lifestyle tokens are recorded at assessment date; ICD-10 codes at first-occurrence date.

### `panel.yaml`

Defines biomarker modalities: for each modality, lists UKB field IDs (with feature names) and available visits. See the table in the Biomarker Extraction section above.

### `family_hx_coding.yaml`

Maps UKB coding scheme 1010 integer values to token IDs (1–13). Source: https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id=1010

### `omop_panel.yaml`

Maps biomarker modalities to OMOP concept IDs for cross-cohort compatibility (e.g. with FinnGen). Generated by `omop_map.py` from `all_concepts_numeric_prio.csv`. Covers blood (`wbc`, `lipid`, `lft`, `renal`, etc.) and urine (`nak`, `creat`, `albu`) biomarkers.

---

## OMOP Mapping (`omop_map.py`)

Maps UKB blood and urine biomarker field IDs to OMOP concept IDs. Reads the field-to-OMOP mapping from `$DELPHI_DATA_DIR/ukb/all_concepts_numeric_prio.csv` and applies it to the `blood` and `urine` sections of `panel.yaml`. Output: `dictionary/omop_panel.yaml`.

---

## Shared Utilities (`utils.py`)

| Function | Description |
|----------|-------------|
| `all_ukb_participants()` | Loads `participants/all.bin` — the canonical participant list |
| `load_biomarker_df(fids, visits, preload)` | Loads multiple UKB fields and pivots to long format indexed by `(pid, visit)` |
| `index_by_visit(df, visits)` | Reshapes wide-format visit columns into a `pd.Series` with `MultiIndex(pid, visit)` |
| `build_expansion_pack(...)` | Writes expansion pack binary format; initializes `p2i` from all UKB participants (so every participant has an entry, even those with 0 tokens) |
| `build_biomarker(biomarker_df, features, odir)` | Writes biomarker binary format; filters non-Delphi participants and NaN rows |
| `load_visit(fid, visit_idx)` | Returns a dict mapping participant IDs to a single visit's measurement |
| `_long_assessment_age()` | Cached long-format assessment age series indexed by `(pid, visit)` |

---

## Disk Layout

```
$DELPHI_DATA_DIR/ukb_real_data/
├── data.bin                          # uint32, all core token sequences concatenated
├── time.bin                          # uint32, age in days (parallel to data.bin)
├── p2i.csv                           # pid → start_pos, seq_len
├── tokenizer.yaml                    # base vocabulary
├── participants/
│   ├── all.bin                       # uint32, all participant IDs
│   ├── train_fold.bin                # uint32, training set (80%)
│   └── val_fold.bin                  # uint32, validation set (20%)
├── biomarkers/
│   ├── wbc/
│   │   ├── data.bin                  # float32, feature vectors concatenated flat
│   │   ├── p2i.csv                   # pid, visit, start_pos, seq_len, time
│   │   └── features.yaml            # ordered list of feature names
│   ├── lipid/
│   ├── lft/
│   └── ...
└── expansion_packs/
    ├── self_report_ops/
    │   ├── data.bin                  # uint32, local token IDs
    │   ├── time.bin                  # uint32, timestamps in days
    │   ├── p2i.csv                   # pid → start_pos, seq_len
    │   └── tokenizer.yaml           # local vocabulary
    ├── prescriptions_hx/
    ├── family_hx/
    ├── fsf/
    ├── prescriptions/
    └── ops/
```

---

## File Reference

| File | Purpose |
|------|---------|
| `data/ukb_scripts/gather_core.py` | Core event extraction (RAP/PySpark) |
| `data/ukb_scripts/gather.py` | Biomarker extraction |
| `data/ukb_scripts/gather_family_history.py` | Family history expansion pack |
| `data/ukb_scripts/gather_female_specific_factors.py` | Female-specific factors expansion pack |
| `data/ukb_scripts/gather_medications.py` | Self-reported medications expansion pack |
| `data/ukb_scripts/gather_operations.py` | Self-reported operations expansion pack |
| `data/ukb_scripts/gather_summary_operations.py` | Hospital operations (OPCS-4) expansion pack |
| `data/ukb_scripts/gather_prescriptions.py` | GP prescriptions expansion pack |
| `data/ukb_scripts/omop_map.py` | OMOP concept mapping |
| `data/ukb_scripts/utils.py` | Shared utilities (build functions, data loading) |
| `data/ukb_scripts/utils_codon.py` | Codon platform implementation |
| `data/ukb_scripts/utils_rap.py` | RAP platform implementation |
| `data/ukb_scripts/dictionary/tokenizer.yaml` | Base vocabulary |
| `data/ukb_scripts/dictionary/panel.yaml` | Biomarker modality definitions |
| `data/ukb_scripts/dictionary/family_hx_coding.yaml` | Family history coding map |
| `data/ukb_scripts/dictionary/omop_panel.yaml` | OMOP concept IDs |
| `delphi/env.py` | Platform detection and data directory resolution |
