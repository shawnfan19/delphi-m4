from pathlib import Path

import pytest
from pytest import Config, FixtureRequest, Metafunc, Parser

from delphi.env import DELPHI_DATA_DIR


def pytest_addoption(parser: Parser):

    parser.addoption(
        "--dataset",
        action="store",
        default="ukb_real_data",
        help="directory to data to be tested",
    )


def get_dataset_dir(config: Config):

    dataset = config.getoption("--dataset")

    return Path(DELPHI_DATA_DIR) / str(dataset)


@pytest.fixture
def dataset_dir(request: FixtureRequest):

    return get_dataset_dir(request.config)


def pytest_generate_tests(metafunc: Metafunc):
    if "biomarker" in metafunc.fixturenames:
        biomarkers_dir = get_dataset_dir(metafunc.config) / "biomarkers"
        if not biomarkers_dir.is_dir():
            pytest.fail(f"no biomarkers/ dir at {biomarkers_dir}")
        biomarkers = sorted(
            p.name
            for p in biomarkers_dir.iterdir()
            if p.is_dir() and (p / "data.bin").exists()
        )
        metafunc.parametrize("biomarker", biomarkers)
