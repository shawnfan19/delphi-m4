## MIMIC-IV Preprocessing

### Source Data

MIMIC-IV v3.1 hosp module at `/hps/nobackup/birney/users/sfan/delphi-data/physionet.org/files/mimiciv/3.1/hosp/`.

Documentation: `/hps/nobackup/birney/users/sfan/mimic-website/content/en/docs/IV/modules/hosp/`

### Pipeline

Scripts in `data/mimic_scripts/`:

| Script | Purpose |
|--------|---------|
| `build_events.py` | Extract diagnoses, procedures, gender, death → flat binary files |
| `stats.py` | Compute and print dataset statistics |

**Output directory**: `/hps/nobackup/birney/users/sfan/delphi-data/mimic/`

Output format matches UKB (see `AGENTS/DATA.md`): `data.bin`, `time.bin`, `p2i.csv`, `tokenizer.yaml`, `participants/{all,train_fold,val_fold}.bin`.

### Configuration Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--no-procedures` | off | Exclude ICD procedure codes |
| `--no-death` | off | Exclude death tokens |

### Processing Steps

1. Load `patients.csv.gz`, `admissions.csv.gz`, `diagnoses_icd.csv.gz`, `procedures_icd.csv.gz`
2. Convert ICD-9 → ICD-10 via CMS 2018 GEMs crosswalks (cached in `data/mimic_scripts/cache/`)
   - Diagnoses: ICD-9-CM → ICD-10-CM (separate GEMs download)
   - Procedures: ICD-9-PCS → ICD-10-PCS (separate GEMs download)
3. Truncate all ICD-10 codes to 3 characters, lowercase
   - Diagnosis tokens: bare codes (e.g. `i10`, `e78`)
   - Procedure tokens: `px_` prefix (e.g. `px_5a1`, `px_02h`) to avoid collision with diagnosis codes
4. Compute age in days: `anchor_year - anchor_age` → shifted birth year → `(date - birth_date).days`
   - Diagnoses use `admittime` from admissions table
   - Procedures use `chartdate` directly
5. Deduplicate: one occurrence per (subject_id, token), keeping earliest
6. Add gender tokens (`female`/`male`) at age 0
7. Add death tokens from `patients.dod` (age at death)
8. Write binary files + 80/20 train/val split (seed 42)

### Dataset Statistics (all features enabled)

| Metric | Value |
|--------|-------|
| Patients | 223,340 |
| Total tokens | 4,528,637 |
| Vocabulary | 2,477 (1,757 diagnosis + 715 procedure + 5 special) |
| Gender | 52.7% F, 47.3% M |
| Deaths | 36,876 (16.5% of patients) |
| Diagnosis ICD-9 conversion rate | 99.6% |
| Procedure ICD-9 conversion rate | 99.9% |

**Tokens per patient**: mean 20.3, median 15, P95 57, max 251

**Trajectory span** (disease events only): mean 1.5 years, median 0 years, P90 5.9 years, max 15.6 years. 36.5% of patients have zero span (single admission).

### Tokenizer Layout

| Range | Content |
|-------|---------|
| 0–1 | padding, no_event |
| 2–4 | female, male, death |
| 5–1761 | ICD-10-CM diagnosis codes (3-char, e.g. `a01_(typhoid...)`) |
| 1762–2476 | ICD-10-PCS procedure codes (3-char with `px_` prefix, e.g. `px_001_(cns,_bypass)`) |

### Key Differences from UKB

- MIMIC is acute-care hospital data; trajectories are much shorter than UKB's longitudinal primary-care records
- No lifestyle tokens (BMI, smoking, alcohol)
- ~141K patients excluded (no diagnosis codes in hosp module)
- Age approximated to Jan 1 of birth year (exact birth date not available in MIMIC)
- Tokenizer is standalone (not shared with UKB), but uses the same 3-char ICD-10 format for diagnoses
