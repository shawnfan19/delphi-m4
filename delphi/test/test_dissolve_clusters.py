"""`dissolve_clusters` breaks same-timestamp diagnosis clusters into an ordered,
dx-delimited micro-sequence; `pack_clusters` is its inverse.

These pin down the contract the generation/eval round-trip relies on:
  - death (a dissolved token) lands as the LAST disease event in its cluster, but
    strictly BEFORE that cluster's dx anchor;
  - the dissolve -> pack round-trip re-collocates death with its cluster;
  - non-ignored output times are strictly increasing (no residual ties);
  - a too-small preceding gap warns (and never pushes a token forward past its dx);
  - empty input is a no-op.

`delphi.data.utils` imports only numpy, so this runs without any dataset backend.
"""

import numpy as np
import pytest

from delphi.data.utils import dissolve_clusters, pack_clusters

WHITELIST = np.array([0, 1])  # pad, no_event
DX_TOKEN = 99
DEATH_TOKEN = 1269
IGNORE_TOKENS = np.array([0])
EPSILON = 0.01


def _dissolve(x, t, seed=0):
    return dissolve_clusters(
        np.asarray(x, dtype=np.int64),
        np.asarray(t, dtype=np.float64),
        np.random.default_rng(seed),
        whitelist=WHITELIST,
        dx_token=DX_TOKEN,
        death_token=DEATH_TOKEN,
        ignore_tokens=IGNORE_TOKENS,
        epsilon=EPSILON,
    )


def test_death_is_last_disease_before_dx():
    # disease 10 @100; disease 11 + death tied @200 (no whitelist at the death time)
    x, t = _dissolve([10, 11, DEATH_TOKEN], [100.0, 200.0, 200.0])

    assert x.dtype == np.uint32 and t.dtype == np.float32
    # death appears exactly once
    assert int((x == DEATH_TOKEN).sum()) == 1
    i_death = int(np.where(x == DEATH_TOKEN)[0][0])
    # the token immediately following death is the cluster's dx anchor
    assert x[i_death + 1] == DX_TOKEN
    # death is after the other disease in its cluster (11) ...
    i_11 = int(np.where(x == 11)[0][0])
    assert t[i_11] < t[i_death]
    # ... and strictly before its dx anchor
    assert t[i_death] < t[i_death + 1]
    # the sequence ends on a dx delimiter
    assert x[-1] == DX_TOKEN


def test_dissolve_pack_round_trip_recolocates_death():
    x, t = _dissolve([10, 11, DEATH_TOKEN], [100.0, 200.0, 200.0])

    # pack_clusters is batched (B, L); feed copies since it mutates in place
    packed_x, packed_t = pack_clusters(
        x[None, :].copy(), t[None, :].copy(), WHITELIST, dx_token=DX_TOKEN
    )
    packed_x, packed_t = packed_x[0], packed_t[0]

    # dx anchors are removed (zeroed, pushed to -1e4)
    assert int((packed_x == DX_TOKEN).sum()) == 0
    assert int((packed_t == -1e4).sum()) == 2
    # death and disease 11 are snapped back onto the same (cluster) timestamp
    t_death = packed_t[packed_x == DEATH_TOKEN]
    t_11 = packed_t[packed_x == 11]
    assert t_death.size == 1 and t_11.size == 1
    assert t_death[0] == t_11[0]


def test_no_residual_ties_for_non_ignored_tokens():
    # a 3-way tie at 100 plus an earlier event
    _, t = _dissolve([10, 11, 12, 13], [50.0, 100.0, 100.0, 100.0])

    # no ignore tokens present -> every gap is floored to >= epsilon
    assert np.all(np.diff(t) > 0)
    assert len(np.unique(t)) == len(t)


def test_small_gap_warns_and_does_not_jump_forward():
    # two distinct dissolved tokens < 2*epsilon apart triggers the compression path
    with pytest.warns(UserWarning, match=r"2\*epsilon"):
        _, t = _dissolve([10, 11], [100.0, 100.005])

    assert np.all(np.isfinite(t))
    # order preserved, no forward jump past the anchor
    assert np.all(np.diff(t) >= 0)


def test_empty_input_is_noop():
    x, t = _dissolve([], [])
    assert x.size == 0 and t.size == 0
    assert x.dtype == np.uint32 and t.dtype == np.float32
