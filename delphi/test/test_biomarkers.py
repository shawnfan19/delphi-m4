import os

import numpy as np
import pandas as pd
import pytest

from delphi.multimodal import Modality

all_biomarkers = [modality.lower() for modality in Modality.__members__]


def has_required_columns(p2i: pd.DataFrame) -> bool:

    required_columns = {"pid", "visit", "start_pos", "seq_len", "time"}
    return required_columns.issubset(p2i.columns)


def has_all_participants(p2i: pd.DataFrame, pids: np.ndarray) -> bool:

    return bool(np.isin(p2i["pid"].astype(int).to_numpy(), pids).all())


def data_is_1d(data: np.ndarray) -> bool:
    return data.ndim == 1


def no_nan_data(data: np.ndarray) -> bool:
    return not np.isnan(data).any()


def no_empty_data(p2i: pd.DataFrame) -> bool:
    return bool((p2i["seq_len"] > 0).all())


def total_dimensions_match(p2i: pd.DataFrame, data: np.ndarray) -> bool:

    total_seq_len = p2i["seq_len"].sum()
    return data.size == total_seq_len


def no_duplicate_start_pos(p2i: pd.DataFrame) -> bool:

    start_pos = p2i["start_pos"].to_numpy()
    nonzero_start_pos = start_pos[start_pos != 0]
    is_unique = len(nonzero_start_pos) == len(set(nonzero_start_pos))

    return is_unique


@pytest.mark.parametrize("biomarker", all_biomarkers)
def test_biomarkers(dataset_dir, all_participants, biomarker):
    biomarker_path = os.path.join(dataset_dir, "biomarkers", biomarker)

    data = np.fromfile(os.path.join(biomarker_path, "data.bin"), dtype=np.float32)
    p2i = pd.read_csv(os.path.join(biomarker_path, "p2i.csv"))

    assert has_required_columns(p2i=p2i)
    assert has_all_participants(p2i=p2i, pids=all_participants)
    assert data_is_1d(data=data)
    assert no_nan_data(data=data)
    assert no_empty_data(p2i=p2i)
    assert total_dimensions_match(p2i=p2i, data=data)
    assert no_duplicate_start_pos(p2i=p2i)
