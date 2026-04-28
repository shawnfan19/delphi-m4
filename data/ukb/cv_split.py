#!/usr/bin/env python3
"""
Create 5-fold cross-validation splits from existing train/val participant lists.

Fold 0 is the pre-existing split (train_fold.bin / val_fold.bin).
This script creates folds 1–4 by partitioning train_fold.bin into 4 equal
parts and rotating the held-out set, keeping val_fold.bin in the training
split for folds 1–4.

Output (written to the same participants directory):
    train_fold_1.bin  val_fold_1.bin
    train_fold_2.bin  val_fold_2.bin
    train_fold_3.bin  val_fold_3.bin
    train_fold_4.bin  val_fold_4.bin
"""

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participants-dir",
        type=Path,
        default=Path(os.environ["DELPHI_DATA_DIR"]) / "ukb_real_data" / "participants",
        help="Directory containing train_fold.bin and val_fold.bin (default: $DELPHI_DATA_DIR/ukb_real_data/participants)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling train_fold.bin before partitioning (default: 42)",
    )
    args = parser.parse_args()

    participants_dir: Path = args.participants_dir
    train_path = participants_dir / "train_fold.bin"
    val_path = participants_dir / "val_fold.bin"

    # Load existing fold 0
    train0 = np.fromfile(train_path, dtype=np.uint32)
    val0 = np.fromfile(val_path, dtype=np.uint32)

    n_train = len(train0)
    n_val = len(val0)
    total = n_train + n_val

    print(f"Fold 0 (existing): train={n_train:,}  val={n_val:,}  total={total:,}")

    # Shuffle train0 participants, then split into 4 equal parts
    rng = np.random.default_rng(args.seed)
    shuffled = train0.copy()
    rng.shuffle(shuffled)

    # np.array_split handles unequal sizes gracefully (first parts get one extra)
    parts = np.array_split(shuffled, 4)

    for i, part in enumerate(parts, start=1):
        print(f"  part {i}: {len(part):,} participants")

    print()

    for k in range(1, 5):
        val_k = parts[k - 1]
        # train = val_fold.bin (fold 0's val) + the other 3 parts of train_fold.bin
        train_k = np.concatenate([val0] + [parts[j] for j in range(4) if j != k - 1])

        out_train = participants_dir / f"train_fold_{k}.bin"
        out_val = participants_dir / f"val_fold_{k}.bin"

        train_k.tofile(out_train)
        val_k.tofile(out_val)

        print(
            f"Fold {k}: train={len(train_k):,}  val={len(val_k):,}"
            f"  → {out_train.name}, {out_val.name}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
