import os
from pathlib import Path

from delphi.env import DELPHI_DATA_READ


def detect_dataset() -> str:
    """Return 'ukb' or 'aou' based on which dataset subdir exists under DELPHI_DATA_READ.

    Match rule: any first-level subdirectory whose name (lowercased) starts with
    'ukb' counts as UKB; same for 'aou'. Prefix rather than substring so the
    current 'aou_uk' dir resolves to AoU rather than colliding with UKB.
    """
    base = Path(DELPHI_DATA_READ)
    if not base.is_dir():
        raise RuntimeError(f"DELPHI_DATA_READ is not a directory: {base}")
    has_ukb = any(
        p.is_dir() and p.name.lower().startswith("ukb") for p in base.iterdir()
    )
    has_aou = any(
        p.is_dir() and p.name.lower().startswith("aou") for p in base.iterdir()
    )
    if has_ukb and not has_aou:
        return "ukb"
    if has_aou and not has_ukb:
        return "aou"
    if has_ukb and has_aou:
        raise RuntimeError(
            f"both UKB and AoU dirs present at {base}; "
            "set DELPHI_DATASET=ukb|aou to disambiguate"
        )
    raise RuntimeError(f"no UKB or AoU dataset dir found under {base}")


def multimodal_reader_cls():
    """Return the MultimodalReader class for the active dataset.

    Honors DELPHI_DATASET env var override; otherwise auto-detects.
    """
    dataset = os.environ.get("DELPHI_DATASET") or detect_dataset()
    if dataset == "ukb":
        from delphi.data.ukb import MultimodalUKBReader

        return MultimodalUKBReader
    if dataset == "aou":
        from delphi.data.aou import MultimodalAOUReader

        return MultimodalAOUReader
    raise ValueError(f"unknown dataset: {dataset!r} (expected 'ukb' or 'aou')")
