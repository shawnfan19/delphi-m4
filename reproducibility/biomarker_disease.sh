#!/usr/bin/env bash
# Sweep plot/data/biomarker_distribution.py over a curated set of
# (disease, biomarker_feature) pairs on the active dataset.
#
# Each pair produces one PNG under
# results/biomarker_disease/{dataset_name}/{disease_short}_{feature}.png
# with two subplots: all-cases vs cases-where-biomarker-was-measured-pre-
# diagnosis.
#
# Dataset is auto-detected by delphi.data.auto.detect_dataset() — any one
# compute platform hosts a single dataset (UKB or AoU). Re-run this script
# on the other platform to fill in the comparison.
#
# Pairs are listed in the co-located biomarker_disease.yaml. The shell reads
# the YAML via inline Python — it does not accept the YAML as a CLI arg, but
# the YAML is the single source of truth for the pair list (edit there to
# extend coverage).
set -uo pipefail

cd "$(dirname "$0")/.."   # repo root

YAML="reproducibility/biomarker_disease.yaml"

# Expand the YAML into TAB-separated (disease, feature) lines.
PAIRS=$(python - <<EOF
import yaml
with open("$YAML") as f:
    data = yaml.safe_load(f)
for disease, features in data.items():
    for feature in features:
        print(f"{disease}\t{feature}")
EOF
)

while IFS=$'\t' read -r DISEASE FEATURE; do
    echo "=== $DISEASE / $FEATURE ==="
    python plot/data/biomarker_distribution.py \
        "disease=$DISEASE" "feature=$FEATURE"
done <<< "$PAIRS"
