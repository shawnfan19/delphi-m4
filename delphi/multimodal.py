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
    biomarkers, expansion_packs = list(), list()
    all_biomarkers = [m.name for m in Modality]
    for modality in panel:
        if modality.upper() in all_biomarkers:
            biomarkers.append(modality)
        else:
            expansion_packs.append(modality)
    return biomarkers or None, expansion_packs or None, Path(path).stem
