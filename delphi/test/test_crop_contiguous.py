"""crop_contiguous crops the parallel (x, t) arrays to a contiguous window.

Pins the fixed two-array contract its sole caller (TokenTransform) relies on:
a no-op when already within block_size, and a same-slice crop of both arrays
otherwise (left/right deterministic; random stays in-range and contiguous).
"""

import numpy as np

from delphi.data.utils import crop_contiguous

RNG = np.random.default_rng(0)
X = np.arange(10)
T = np.arange(10) * 100  # parallel array, distinct values to track the slice


def test_noop_within_block_size():
    x, t = crop_contiguous(X, T, block_size=10, rng=RNG)
    assert np.array_equal(x, X) and np.array_equal(t, T)


def test_left_keeps_prefix():
    x, t = crop_contiguous(X, T, block_size=4, rng=RNG, mode="left")
    assert x.tolist() == [0, 1, 2, 3] and t.tolist() == [0, 100, 200, 300]


def test_right_keeps_suffix():
    x, t = crop_contiguous(X, T, block_size=4, rng=RNG, mode="right")
    assert x.tolist() == [6, 7, 8, 9] and t.tolist() == [600, 700, 800, 900]


def test_random_is_contiguous_and_aligned():
    x, t = crop_contiguous(
        X, T, block_size=4, rng=np.random.default_rng(1), mode="random"
    )
    assert len(x) == len(t) == 4
    assert np.array_equal(np.diff(x), np.ones(3))  # contiguous block
    assert np.array_equal(t, x * 100)  # both arrays sliced identically


if __name__ == "__main__":
    test_noop_within_block_size()
    test_left_keeps_prefix()
    test_right_keeps_suffix()
    test_random_is_contiguous_and_aligned()
    print("ok")
