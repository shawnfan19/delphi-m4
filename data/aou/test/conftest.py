from pathlib import Path

import pytest
import yaml
from cloudpathlib import AnyPath
from pytest import Config, FixtureRequest, Metafunc, Parser

from delphi.env import DELPHI_DATA_DIR


def pytest_addoption(parser: Parser):

    parser.addoption(
        "--dataset",
        action="store",
        default="aou_uk",
        help="directory to data to be tested",
    )


def get_dataset_dir(config: Config):

    dataset = config.getoption("--dataset")

    return AnyPath(DELPHI_DATA_DIR) / str(dataset)


@pytest.fixture
def dataset_dir(request: FixtureRequest):

    return get_dataset_dir(request.config)


@pytest.fixture
def panel_config():
    path = Path(__file__).resolve().parent.parent.parent / "panel" / "aou.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def biomarker_config():
    path = Path(__file__).resolve().parent.parent.parent / "biomarker.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pytest_generate_tests(metafunc: Metafunc):
    if "panel" in metafunc.fixturenames:
        biomarkers_dir = get_dataset_dir(metafunc.config) / "biomarkers"
        if not biomarkers_dir.is_dir():
            pytest.fail(f"no biomarkers/ dir at {biomarkers_dir}")
        panels = sorted(
            p.name
            for p in biomarkers_dir.iterdir()
            if p.is_dir() and (p / "data.parquet").exists()
        )
        metafunc.parametrize("panel", panels)
