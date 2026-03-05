#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # MIMIC-IV Dataset Statistics
#
# Computes and prints statistics for the processed MIMIC-IV event sequences.
# Run after `build_events.py`.

# %%
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    default="/hps/nobackup/birney/users/sfan/delphi-data/mimic",
)
args = parser.parse_args([])

DATA_DIR = Path(args.data_dir)

# %% [markdown]
# ## Load processed data

# %%
p2i = pd.read_csv(DATA_DIR / "p2i.csv")
data = np.fromfile(DATA_DIR / "data.bin", dtype=np.uint32)
time = np.fromfile(DATA_DIR / "time.bin", dtype=np.uint32)

with open(DATA_DIR / "tokenizer.yaml") as f:
    tokenizer = yaml.safe_load(f)
id2token = {v: k for k, v in tokenizer.items()}

DAYS_PER_YEAR = 365.25

print(f"Total tokens: {len(data)}")
print(f"Total patients: {len(p2i)}")
print(f"Vocabulary size: {len(tokenizer)}")

# %% [markdown]
# ## Tokens per patient

# %%
seq_lens = p2i["seq_len"].values

print("\n--- Tokens per patient ---")
print(f"  Mean:   {seq_lens.mean():.1f}")
print(f"  Median: {np.median(seq_lens):.1f}")
print(f"  Min:    {seq_lens.min()}")
print(f"  Max:    {seq_lens.max()}")
print(f"  Std:    {seq_lens.std():.1f}")
for pct in [25, 50, 75, 90, 95, 99]:
    print(f"  P{pct}:    {np.percentile(seq_lens, pct):.0f}")

# %% [markdown]
# ## Trajectory span (time from first to last disease event)
#
# Excludes the gender token (age 0) to show the actual span of medical events.

# %%
female_id = tokenizer.get("female", None)
male_id = tokenizer.get("male", None)
gender_ids = {female_id, male_id} - {None}

spans_days = []
for _, row in p2i.iterrows():
    start = row["start_pos"]
    length = row["seq_len"]
    d = data[start : start + length]
    t = time[start : start + length]
    # Exclude gender tokens
    mask = ~np.isin(d, list(gender_ids))
    t = t[mask]
    if len(t) <= 1:
        spans_days.append(0)
        continue
    spans_days.append(int(t[-1]) - int(t[0]))

spans_days = np.array(spans_days)
spans_years = spans_days / DAYS_PER_YEAR

print("\n--- Trajectory span (years) ---")
print(f"  Mean:   {spans_years.mean():.1f}")
print(f"  Median: {np.median(spans_years):.1f}")
print(f"  Min:    {spans_years.min():.1f}")
print(f"  Max:    {spans_years.max():.1f}")
print(f"  Std:    {spans_years.std():.1f}")
for pct in [25, 50, 75, 90, 95, 99]:
    print(f"  P{pct}:    {np.percentile(spans_years, pct):.1f}")

# Zero-span patients (single event or all events on same day)
n_zero_span = (spans_days == 0).sum()
print(
    f"\n  Zero-span patients: {n_zero_span} ({n_zero_span / len(spans_days) * 100:.1f}%)"
)

# %% [markdown]
# ## Gender distribution

# %%
n_female = (data == female_id).sum() if female_id is not None else 0
n_male = (data == male_id).sum() if male_id is not None else 0
print(f"\n--- Gender ---")
print(f"  Female: {n_female} ({n_female / len(p2i) * 100:.1f}%)")
print(f"  Male:   {n_male} ({n_male / len(p2i) * 100:.1f}%)")

# %% [markdown]
# ## Most common disease tokens

# %%
# Exclude special tokens
special_ids = {tokenizer.get(k, -1) for k in ["padding", "no_event", "female", "male"]}
disease_mask = ~np.isin(data, list(special_ids))
disease_tokens = data[disease_mask]

token_counts = pd.Series(disease_tokens).value_counts()
print(f"\n--- Top 20 disease tokens ---")
for tid, count in token_counts.head(20).items():
    name = id2token.get(tid, f"unknown_{tid}")
    print(f"  {name}: {count}")

# %% [markdown]
# ## Age distribution at events

# %%
# Exclude gender tokens (age=0 placeholders)
event_mask = data != female_id
if male_id is not None:
    event_mask &= data != male_id
event_ages = time[event_mask] / DAYS_PER_YEAR

print(f"\n--- Age at events (years) ---")
print(f"  Mean:   {event_ages.mean():.1f}")
print(f"  Median: {np.median(event_ages):.1f}")
print(f"  Min:    {event_ages.min():.1f}")
print(f"  Max:    {event_ages.max():.1f}")
for pct in [25, 50, 75, 90, 95]:
    print(f"  P{pct}:    {np.percentile(event_ages, pct):.1f}")

# %%
