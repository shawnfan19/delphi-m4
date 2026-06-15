#!/usr/bin/env bash
# Sweep apps/explain_grad_lg.py over a curated set of biomarker -> disease pairs,
# saving the overview + stratified-context figures and teeing each run's stdout
# (R2 context-beyond-value + top context coefficients) to a per-pair log.
#
# Runs sequentially on purpose: the cbc / panel artifacts decompress to several GB
# of jacobians, so parallel loads would be memory-heavy and all share one GPU.
#
# All pairs read the existing saliency-*.npz artifacts under
# cross-cohort/blood+urine/ (every file already carries jacobians for all 1257
# disease targets, so probing a new disease only means selecting a different
# target column -- no model re-run). Response is the log-scale saliency
# d(log-intensity)/d value; pass scale=intensity to switch.
set -uo pipefail

# DELPHI_CKPT_DIR is resolved inside the python script (env.py); the shell only
# needs it to place logs, so fall back to DELPHI_CKPT_WRITE / the known location.
CKPT_DIR="${DELPHI_CKPT_DIR:-${DELPHI_CKPT_WRITE:-/hps/nobackup/birney/users/sfan/delphi-ckpt}}"
DIR="cross-cohort/blood+urine"          # relative to DELPHI_CKPT_DIR
FIGDIR="$DIR/figures"
LOGDIR="$CKPT_DIR/$DIR/figures/logs"
SCALE="${SCALE:-log}"
ONLY_STAGE="${ONLY_STAGE:-}"   # if set (e.g. 4), run only pairs whose stage field == this
mkdir -p "$LOGDIR"

# spec = file|feature|target|stage|note
#
# ===========================================================================
# ARCHIVED — the original 24 pairs (pre-improved-disease filter), kept for
# reference. Commented out because explaining a biomarker's saliency for a
# disease whose prediction did NOT improve when biomarkers were added is
# unfounded. 11 of these target diseases outside the top-20 Δ c-index list
# (results/blood/improved.yaml): i25, n40, k76, m81, m05, i10, i50, c61, c50.
# Re-enable any line individually if you want to inspect it anyway.
# ---------------------------------------------------------------------------
# "saliency-renal_panel.npz|creatinine|n18_(chronic_renal_failure)|1|creatinine->CKD"
# "saliency-cbc.npz|haemoglobin_concentration|d50_(iron_deficiency_anaemia)|1|Hb->IDA"
# "saliency-glycated_haemoglobin.npz|glycated_haemoglobin|e11_(non-insulin-dependent_diabetes_mellitus)|1|HbA1c->T2D"
# "saliency-gamma_glutamyltransferase.npz|gamma_glutamyltransferase|k70_(alcoholic_liver_disease)|1|GGT->ALD"
# "saliency-urate.npz|urate|m10_(gout)|2|urate->gout"
# "saliency-cystatin_c.npz|cystatin_c|n18_(chronic_renal_failure)|2|cysC->CKD (control)"
# "saliency-lipid_panel.npz|ldl_direct|i25_(chronic_ischaemic_heart_disease)|2|LDL->CHD  [DROPPED: i25 not improved]"
# "saliency-lipid_panel.npz|hdl|i25_(chronic_ischaemic_heart_disease)|2|HDL->CHD  [DROPPED: i25 not improved]"
# "saliency-testosterone.npz|testosterone|n40_(hyperplasia_of_prostate)|2|T->BPH  [DROPPED: n40 not improved]"
# "saliency-shbg.npz|shbg|e11_(non-insulin-dependent_diabetes_mellitus)|2|SHBG->T2D"
# "saliency-lft_panel.npz|aspartate_aminotransferase|k70_(alcoholic_liver_disease)|2|AST->ALD"
# "saliency-lft_panel.npz|alanine_aminotransferase|k76_(other_diseases_of_liver)|2|ALT->NAFLD  [DROPPED: k76 not improved]"
# "saliency-vitamin_d.npz|vitamin_d|m81_(osteoporosis_without_pathological_fracture)|2|vitD->osteoporosis  [DROPPED: m81 not improved]"
# "saliency-rheumatoid_factor.npz|rheumatoid_factor|m05_(seropositive_rheumatoid_arthritis)|2|RF->RA  [DROPPED: m05 not improved]"
# "saliency-cbc.npz|haemoglobin_concentration|d64_(other_anaemias)|2|Hb->other anaemia"
# "saliency-urine_microalbumin.npz|urine_microalbumin|n18_(chronic_renal_failure)|2|microalbumin->CKD"
# "saliency-urate.npz|urate|i10_(essential_(primary)_hypertension)|3|urate->HTN  [DROPPED: i10 not improved]"
# "saliency-gamma_glutamyltransferase.npz|gamma_glutamyltransferase|e11_(non-insulin-dependent_diabetes_mellitus)|3|GGT->T2D"
# "saliency-cbc.npz|mean_corpuscular_volume|k70_(alcoholic_liver_disease)|3|MCV->ALD"
# "saliency-cbc.npz|red_blood_cell_distribution_width|i50_(heart_failure)|3|RDW->HF  [DROPPED: i50 not improved]"
# "saliency-igf1.npz|igf1|c61_malignant_neoplasm_of_prostate|3|IGF1->prostate ca  [DROPPED: c61 not improved]"
# "saliency-shbg.npz|shbg|c50_malignant_neoplasm_of_breast|3|SHBG->breast ca  [DROPPED: c50 not improved]"
# "saliency-lipid_panel.npz|triglycerides|e11_(non-insulin-dependent_diabetes_mellitus)|3|TG->T2D"
# "saliency-cystatin_c.npz|cystatin_c|i50_(heart_failure)|3|cysC->HF  [DROPPED: i50 not improved]"
#
# ===========================================================================
# ACTIVE — improved-disease pairs only. Every target is in the top-20 by
# Δ c-index (baseline -> blood+urine); see results/blood/improved.yaml.
# stage 1-3 = survivors of the original 24 whose target disease improved;
# stage 4 = new probes covering the other improved diseases via their driving
# biomarker. (d75 / c67 bladder / k26 ulcer improved but have no clean blood
# marker in these panels, so they are not probed.)
# ===========================================================================
PAIRS=(
  # -- e21_(hyperparathyroidism_and_other_disorders_of_parathyroid_gland) --
  "saliency-renal_panel.npz|calcium|e21_(hyperparathyroidism_and_other_disorders_of_parathyroid_gland)|dd|spec3.6 signed+0.19 (data-driven)"
  "saliency-renal_panel.npz|phosphate|e21_(hyperparathyroidism_and_other_disorders_of_parathyroid_gland)|dd|spec2.8 signed-0.15 (data-driven)"
  # -- c91_lymphoid_leukaemia --
  "saliency-cbc.npz|eosinophill_count|c91_lymphoid_leukaemia|dd|spec2.8 signed-0.09 (data-driven)"
  "saliency-cbc.npz|eosinophill_percentage|c91_lymphoid_leukaemia|dd|spec2.7 signed-0.10 (data-driven)"
  # -- m10_(gout) --
  "saliency-urate.npz|urate|m10_(gout)|dd|spec4.9 signed+0.56 (data-driven)"
  "saliency-shbg.npz|shbg|m10_(gout)|dd|spec2.0 signed-0.12 (data-driven)"
  # -- e14_(unspecified_diabetes_mellitus) --
  "saliency-glycated_haemoglobin.npz|glycated_haemoglobin|e14_(unspecified_diabetes_mellitus)|dd|spec3.6 signed+0.69 (data-driven)"
  "saliency-renal_panel.npz|glucose|e14_(unspecified_diabetes_mellitus)|dd|spec2.2 signed+0.15 (data-driven)"
  # -- e80_(disorders_of_porphyrin_and_bilirubin_metabolism) --
  "saliency-cbc.npz|red_blood_cell_count|e80_(disorders_of_porphyrin_and_bilirubin_metabolism)|dd|spec2.4 signed+0.12 (data-driven)"
  "saliency-cbc.npz|haemoglobin_concentration|e80_(disorders_of_porphyrin_and_bilirubin_metabolism)|dd|spec2.2 signed+0.17 (data-driven)"
  # -- n18_(chronic_renal_failure) --
  "saliency-renal_panel.npz|creatinine|n18_(chronic_renal_failure)|dd|spec4.8 signed+0.92 (data-driven)"
  "saliency-cystatin_c.npz|cystatin_c|n18_(chronic_renal_failure)|dd|spec3.5 signed+0.44 (data-driven)"
  # -- e78_(disorders_of_lipoprotein_metabolism_and_other_lipidaemias) --
  "saliency-apolipoprotein_b.npz|apolipoprotein_b|e78_(disorders_of_lipoprotein_metabolism_and_other_lipidaemias)|dd|spec4.2 signed+0.25 (data-driven)"
  "saliency-lipid_panel.npz|cholesterol|e78_(disorders_of_lipoprotein_metabolism_and_other_lipidaemias)|dd|spec3.0 signed+0.12 (data-driven)"
  # -- k74_(fibrosis_and_cirrhosis_of_liver) --
  "saliency-igf1.npz|igf1|k74_(fibrosis_and_cirrhosis_of_liver)|dd|spec3.0 signed-0.15 (data-driven)"
  "saliency-gamma_glutamyltransferase.npz|gamma_glutamyltransferase|k74_(fibrosis_and_cirrhosis_of_liver)|dd|spec2.6 signed+0.45 (data-driven)"
  # -- e11_(non-insulin-dependent_diabetes_mellitus) --
  "saliency-glycated_haemoglobin.npz|glycated_haemoglobin|e11_(non-insulin-dependent_diabetes_mellitus)|dd|spec4.3 signed+0.81 (data-driven)"
  "saliency-renal_panel.npz|glucose|e11_(non-insulin-dependent_diabetes_mellitus)|dd|spec2.7 signed+0.19 (data-driven)"
  # -- d73_(diseases_of_spleen) --
  "saliency-cbc.npz|red_blood_cell_distribution_width|d73_(diseases_of_spleen)|dd|spec1.7 signed+0.17 (data-driven)"
  "saliency-cbc.npz|mean_corpuscular_volume|d73_(diseases_of_spleen)|dd|spec1.2 signed+0.08 (data-driven)"
  # -- d50_(iron_deficiency_anaemia) --
  "saliency-cbc.npz|haematocrit_percentage|d50_(iron_deficiency_anaemia)|dd|spec3.1 signed-0.22 (data-driven)"
  "saliency-cbc.npz|haemoglobin_concentration|d50_(iron_deficiency_anaemia)|dd|spec3.1 signed-0.24 (data-driven)"
  # -- d75_(other_diseases_of_blood_and_blood-forming_organs) --
  "saliency-cbc.npz|mean_corpuscular_haemoglobin|d75_(other_diseases_of_blood_and_blood-forming_organs)|dd|spec2.5 signed+0.14 (data-driven)"
  "saliency-cbc.npz|mean_corpuscular_volume|d75_(other_diseases_of_blood_and_blood-forming_organs)|dd|spec2.4 signed+0.16 (data-driven)"
  # -- k70_(alcoholic_liver_disease) --
  "saliency-gamma_glutamyltransferase.npz|gamma_glutamyltransferase|k70_(alcoholic_liver_disease)|dd|spec3.1 signed+0.55 (data-driven)"
  "saliency-igf1.npz|igf1|k70_(alcoholic_liver_disease)|dd|spec3.1 signed-0.15 (data-driven)"
  # -- j33_(nasal_polyp) --
  "saliency-cbc.npz|eosinophill_percentage|j33_(nasal_polyp)|dd|spec4.3 signed+0.17 (data-driven)"
  "saliency-cbc.npz|eosinophill_count|j33_(nasal_polyp)|dd|spec3.9 signed+0.13 (data-driven)"
  # -- d52_(folate_deficiency_anaemia) --
  "saliency-cbc.npz|red_blood_cell_count|d52_(folate_deficiency_anaemia)|dd|spec2.6 signed-0.13 (data-driven)"
  "saliency-cbc.npz|red_blood_cell_distribution_width|d52_(folate_deficiency_anaemia)|dd|spec2.5 signed+0.26 (data-driven)"
  # -- n19_(unspecified_renal_failure) --
  "saliency-cystatin_c.npz|cystatin_c|n19_(unspecified_renal_failure)|dd|spec2.7 signed+0.34 (data-driven)"
  "saliency-renal_panel.npz|creatinine|n19_(unspecified_renal_failure)|dd|spec2.3 signed+0.43 (data-driven)"
  # -- d64_(other_anaemias) --
  "saliency-cbc.npz|haematocrit_percentage|d64_(other_anaemias)|dd|spec3.1 signed-0.22 (data-driven)"
  "saliency-cbc.npz|haemoglobin_concentration|d64_(other_anaemias)|dd|spec3.1 signed-0.24 (data-driven)"
)

i=0
for spec in "${PAIRS[@]}"; do
  IFS='|' read -r file feature target stage note <<<"$spec"
  if [ -n "$ONLY_STAGE" ] && [ "$stage" != "$ONLY_STAGE" ]; then continue; fi
  i=$((i + 1))
  safe_target=$(echo "$target" | tr -c '0-9a-zA-Z' '_' | sed 's/_*$//')
  log="$LOGDIR/s${stage}__${feature}__${safe_target}__${SCALE}.log"
  echo "[$i/${#PAIRS[@]}] stage $stage | $feature -> $target | $note"
  MPLBACKEND=Agg python apps/explain_grad_lg.py \
    "saliency=$DIR/$file" \
    "feature=$feature" \
    "target=$target" \
    "scale=$SCALE" \
    "figdir=$FIGDIR" >"$log" 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "    FAILED (exit $status) -- see $log"
    tail -5 "$log" | sed 's/^/      /'
  else
    grep -E "context beyond value|R. \(value" "$log" | sed 's/^/      /'
  fi
done
echo "done. figures in $CKPT_DIR/$FIGDIR, logs in $LOGDIR"
