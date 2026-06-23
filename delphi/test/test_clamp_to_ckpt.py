"""`_clamp_to_ckpt` clamps a CLI biomarker/expansion override to the ckpt's set.

Pins the contract the eval apps rely on: None inherits the trained set; an
override is intersected with it (sorted); a non-overlapping override yields the
empty set (a loud warning, not a silent wrong-set eval).
"""

from delphi.experiment import _clamp_to_ckpt


def test_none_inherits_ckpt_set():
    assert _clamp_to_ckpt(None, ["wbc", "lipid"], "biomarkers") == ["wbc", "lipid"]
    # None ckpt value -> empty inherited set
    assert _clamp_to_ckpt(None, None, "biomarkers") == []


def test_override_is_intersected_and_sorted():
    # "renal" is not in the ckpt set -> dropped; result sorted
    out = _clamp_to_ckpt(
        ["renal", "lipid", "wbc"], ["wbc", "lipid", "lft"], "biomarkers"
    )
    assert out == ["lipid", "wbc"]


def test_no_overlap_returns_empty(capsys):
    out = _clamp_to_ckpt(["renal"], ["wbc", "lipid"], "biomarkers")
    assert out == []
    assert "no overlap" in capsys.readouterr().out


if __name__ == "__main__":
    test_none_inherits_ckpt_set()
    test_override_is_intersected_and_sorted()
    # capsys is a pytest fixture; exercise the path without it under __main__
    assert _clamp_to_ckpt(["renal"], ["wbc", "lipid"], "biomarkers") == []
    print("ok")
