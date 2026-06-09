# AoU vs UKB biomarker distribution comparison

## Why this document exists

The Delphi-M4 model was trained on UKB biomarkers and z-scores its biomarker inputs using
**UKB train-set mean and std**. When we evaluate it on AoU the same z-score lookup is
applied — but if AoU biomarker distributions are systematically shifted, the typical AoU
patient lands far from N(0, 1) in the model's input frame.

A bare c-index comparison on AoU showed the biomarker model losing to the
baseline-without-biomarkers (mean Δ c-index −0.0039, biomarker model worse on neoplasms
in particular). The hypothesis we want to confirm or falsify: the gap is driven by
**distribution shift between AoU and UKB biomarker values**, not by a unit-conversion bug.

## Methodology

Both sides use **first occurrence per participant**, matching what the model sees at eval
time (`BiomarkerTransform(first_time_only=True)`):

- `data/ukb/codon/biomarker_stats.py` reads each biomarker straight from the UKB tab
  parquet via `UKBDatabase.load_fid()`. For each participant it back-fills across visit
  instances (instance 0 = init_assess, 1 = repeat, …) and keeps the first non-NaN.
- `data/aou/biomarker_stats.py` runs on the AoU workbench and reads the per-panel
  `data.parquet`. After the panel-level `dropna`, rows are sorted by `(person_id, age_in_days)`
  and deduplicated to one row per `person_id` (earliest visit).

Both scripts emit `stats.csv` with `n, mean, std, median, q25, q75, min, max` per
biomarker. `data/ukb/codon/compare_biomarker_stats.py` joins them and adds:

- `delta_median_in_ukb_sigma = (median_aou - mean_ukb) / std_ukb` — where the typical AoU
  value lands in the model's z-score frame. `|Δ| > 1` ≈ severely out-of-distribution.
- `iqr_ratio = iqr_aou / iqr_ukb` — spread comparison. `>> 1` ⇒ AoU population is much
  wider (e.g., ascertainment toward sicker).

Sorted by `|delta_median_in_ukb_sigma|`. Output:
`$DELPHI_DATA_DIR/biomarker_stats_diff.csv`.

## Top 20 most-shifted biomarkers (first-occurrence)

| biomarker | n_aou | n_ukb | median_aou | median_ukb | Δ in σ | iqr_ratio |
|---|---:|---:|---:|---:|---:|---:|
| **cystatin_c** | 2,449 | 470,635 | 1.27 | 0.89 | **+2.05** | 5.34 |
| **albumin** | 64,563 | 432,173 | 41.0 | 45.20 | **−1.60** | 2.03 |
| **vitamin_d** | 49,614 | 449,783 | 75.0 | 46.8 | **+1.25** | 1.50 |
| **calcium** | 65,270 | 432,024 | 2.270 | 2.376 | **−1.16** | 1.92 |
| **urine_potassium** | 9,010 | 483,784 | 29.1 | 56.9 | **−1.00** | 0.61 |
| MCH_concentration | 61,697 | 479,159 | 33.5 | 34.46 | −0.94 | 1.16 |
| ldl_direct | 62,963 | 469,826 | 2.74 | 3.52 | −0.94 | 1.08 |
| cholesterol | 62,963 | 470,664 | 4.76 | 5.65 | −0.82 | 0.94 |
| oestradiol | 12,878 | 77,665 | 146.8 | 312.5 | −0.73 | 0.69 |
| rheumatoid_factor | 9,900 | 41,972 | 10.5 | 16.7 | −0.71 | 0.26 |
| MCH | 61,697 | 479,162 | 30.1 | 31.5 | −0.70 | 1.30 |
| igf1 | 3,011 | 468,206 | 17.91 | 21.24 | −0.61 | 1.52 |
| apolipoprotein_b | 1,256 | 468,332 | 0.90 | 1.02 | −0.55 | 1.13 |
| haemoglobin_concentration | 61,697 | 479,165 | 13.5 | 14.15 | −0.54 | 1.17 |
| eosinophill_count | 61,697 | 478,299 | 0.10 | 0.14 | −0.54 | 1.00 |
| triglycerides | 62,963 | 470,294 | 1.23 | 1.48 | −0.50 | 0.93 |
| shbg | 7,472 | 428,011 | 38.0 | 45.29 | −0.49 | 1.02 |
| urine_sodium | 11,669 | 483,754 | 56.0 | 68.4 | −0.48 | 1.02 |
| glucose | 65,270 | 431,678 | 5.72 | 4.93 | +0.48 | 3.18 |
| apolipoprotein_a | 1,409 | 429,626 | 1.42 | 1.51 | −0.42 | 1.19 |

5 biomarkers exceed |1σ|. The full sorted list is at
`$DELPHI_DATA_DIR/biomarker_stats_diff.csv`.

## What dropped out between all-occurrence and first-occurrence

Three biomarkers that looked severely shifted on all-occurrence vanished from the top 20
after switching to first-occurrence:

| biomarker | all-occ Δ | first-occ Δ |
|---|---:|---:|
| glycated_haemoglobin | +1.04 | ≈ +0.4 (out of top 20) |
| direct_bilirubin | +1.87 | ≈ +0.4 (out of top 20) |
| creatinine | +0.86 | ≈ +0.3 (out of top 20) |

These shifts were driven by **repeat measurements monitoring an existing condition**
(post-diagnosis HbA1c, hepatobiliary disease workup, CKD follow-up). The model only sees
first-occurrence at eval time, so these are non-issues for transfer.

## Interpretation of the persistent shifts

### Disease-ascertainment, robust to first-occurrence

- **cystatin_c +2.05σ**: the test is ordered specifically when CKD is suspected, so even
  the *first* AoU cystatin_c is enriched for kidney impairment. The model has no way to
  know an AoU patient was sick enough to warrant the test in the first place.
- **albumin −1.60σ**, **glucose +0.48σ (with iqr_ratio 3.2)**: lower albumin and broader
  glucose distribution reflect more chronic illness / inflammation / dysglycemia in the
  AoU lab-tested cohort.

### Therapy-conditioning, robust to first-occurrence

- **ldl_direct −0.94σ**, **cholesterol −0.82σ**, **apolipoprotein_b −0.55σ**,
  **triglycerides −0.50σ**: statin therapy is *chronic*, so first-measurement in the AoU
  data window typically still reflects treated lipids. The model learned "high
  cholesterol → CVD"; on AoU it sees suppressed cholesterol, no longer a reliable signal.

### Population / supplementation / demographic differences

- **vitamin_d +1.25σ**: AoU's first vit-D values are genuinely higher than UKB's. Likely
  US supplementation culture — by the time a patient gets tested, they've usually been on
  replacement therapy.
- **CBC components down** (MCH_conc, MCH, haemoglobin, eosinophill_count, RBC count):
  consistent with more demographically diverse AoU population (higher prevalence of
  haemoglobinopathies and anemia compared to UKB's middle-aged Britons).
- **oestradiol −0.73σ**: AoU oestradiol much lower. Likely different age/sex composition
  of who gets tested (postmenopausal workup, hypogonadism workup in men).

### Indication-specific ordering (a stronger form of ascertainment)

- **urine_potassium −1.00σ (iqr_ratio 0.61)**, **urine_sodium −0.48σ**: not a specimen
  mismatch — UKB's urine fields (30520/30530) are also spot urine, and urine_creatinine
  (which would dilute identically) is well aligned. The story is selection: in US
  clinical practice urine_K is ordered mostly for hypokalemia workup (low-K-clustered
  cohort, narrow IQR), urine_Na for hyponatremia workup; n_aou ≈ 9–12k vs n for
  urine_creatinine ≈ 364k. UKB measured these on every participant at the assessment
  centre, so its distribution reflects dietary variation in the general population
  rather than a low-K or hyponatremia-workup subset.

### Small-σ amplification

- **calcium −1.16σ**: UKB calcium σ is only 0.094 mmol/L (tightly regulated physiology),
  so a 0.10 mmol/L absolute shift looks dramatic in σ units. Real absolute shift is ~5%.

## Implication for the eval gap

The first-occurrence picture is **less alarming than all-occurrence suggested but still
substantial**:

- The biomarker model sees ≥0.5σ shifts on ~16 biomarkers and ≥1σ shifts on 5 — out of
  51 biomarkers in the AoU panel set.
- The remaining shifts are **population/ascertainment/therapy-driven**, not
  preprocessing-fixable. No unit-conversion bug is visible.
- Combined with the asymmetric token gap (AoU lacks UKB's smoking/alcohol lifestyle
  tokens) and the AoU-specific ascertainment bias (sicker patients have *more* biomarker
  tokens), it's plausible that the biomarker module is actively misleading rather than
  just noisy on transfer.

## Next step

The productive fix is to **recompute z-score stats on a held-out AoU calibration subset**
and substitute them in place of UKB train stats inside `BiomarkerTransform.from_ckpt`.
That decouples "is the model architecture transferable" from "are the input stats
transferable", and absorbs the population/therapy shifts that the data layer can't.
