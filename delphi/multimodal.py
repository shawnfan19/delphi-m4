from enum import Enum
from pathlib import Path

import yaml


class Modality(Enum):
    # 0 for padding; 1 for event tokens
    PRS = 2
    WBC = 3
    LIPID = 4
    LFT = 5
    RENAL = 6
    HBA1C = 7
    CRP = 8
    URATE = 9
    CYSC = 10
    APO = 11
    VITD = 12
    DHT = 13
    SHBG = 14
    IGF1 = 15
    NAK = 16
    CREAT = 17
    ALBU = 18
    DIET = 19
    MET = 20
    TELOMERE = 21
    ABDO_FAT_CROSS = 22
    ABDO_FAT_LONG = 23
    NMR = 24
    CHIP_LITE = 25
    CHRS = 26
    PROTEOMICS = 27


def module_name(modality: Modality) -> str:

    module_name = str(modality).split(".")[-1].lower()

    return module_name


def parse_panel(path):
    with open(path) as f:
        panel = yaml.safe_load(f)
    if isinstance(panel, list):
        raise ValueError(
            f"panel {path} uses the old flat-list format; "
            f"convert to {{biomarkers: [...], expansion_packs: [...]}}"
        )
    if not isinstance(panel, dict):
        raise ValueError(f"panel {path} must be a YAML mapping")
    allowed = {"biomarkers", "expansion_packs"}
    unknown = set(panel) - allowed
    if unknown:
        raise ValueError(
            f"unknown panel keys: {sorted(unknown)}; allowed: {sorted(allowed)}"
        )
    for key in allowed:
        if key in panel and panel[key] is not None and not isinstance(panel[key], list):
            raise ValueError(
                f"panel key '{key}' must be a list, got {type(panel[key]).__name__}"
            )
    biomarkers = panel.get("biomarkers") or None
    expansion_packs = panel.get("expansion_packs") or None
    if biomarkers and expansion_packs:
        overlap = sorted(set(biomarkers) & set(expansion_packs))
        if overlap:
            raise ValueError(f"name(s) appear in both lists: {overlap}")
    return biomarkers, expansion_packs, Path(path).stem
