import pytest

from delphi.experiment import match_unique


def test_exact_match_wins_over_substring():
    # "death" is a substring of the other two but an exact match to one choice
    choices = ["death", "foetal_death_unspecified", "o96_death_obstetric"]
    assert match_unique("death", choices) == "death"


def test_substring_match_still_works():
    assert (
        match_unique("pancrea", ["c25_pancreas", "i10_hypertension"]) == "c25_pancreas"
    )


def test_case_insensitive_exact():
    assert match_unique("DEATH", ["death", "foetal_death"]) == "death"


def test_ambiguous_substring_raises():
    with pytest.raises(SystemExit):
        match_unique("itis", ["arthritis", "dermatitis"])


def test_no_match_raises():
    with pytest.raises(SystemExit):
        match_unique("nope", ["arthritis", "dermatitis"])
