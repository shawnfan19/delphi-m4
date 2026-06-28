"""Hand-checked tests for the three cluster/co-occurrence trackers.

Pins the shared ``_flatten_events`` extraction (a ``(row, time)`` pair groups
tokens into one event, all-padding batches are skipped) and the contrasting
co-occurrence semantics: ``TiedEventTracker`` counts same-``(participant, day)``
events, while the participant-level ``CooccurrenceTracker`` counts lifetime
presence — optionally directed via ``before``.
"""

import numpy as np

from delphi.eval.cluster import (
    ClusterStatsTracker,
    CooccurrenceTracker,
    MissingDxTracker,
    TiedEventTracker,
    _flatten_events,
)

# Batched fixtures for the event-keyed trackers.
# row 0: tokens 5 & 7 at day 10 (a cluster of 2); row 1: token 5 at day 10.
TOKENS = np.array([[5, 7, 0], [5, 0, 0]])
TIMES = np.array([[10, 10, 0], [10, 0, 0]])
PAD = np.zeros((2, 3), dtype=int)

# Single-participant fixture for CooccurrenceTracker: 5 repeats, then 7 — on
# different days, so it separates lifetime from tied-event co-occurrence.
SEQ_TOKENS = np.array([5, 5, 7])
SEQ_TIMES = np.array([10, 15, 20])


def test_flatten_events_skips_all_padding():
    assert _flatten_events(PAD, PAD) is None


def test_flatten_events_drops_padding():
    flat_rows, flat_tokens, flat_times, event_keys = _flatten_events(TOKENS, TIMES)
    assert flat_rows.tolist() == [0, 0, 1]
    assert flat_tokens.tolist() == [5, 7, 5]
    assert flat_times.tolist() == [10, 10, 10]
    assert event_keys.tolist() == [[0, 10], [0, 10], [1, 10]]


def test_cluster_stats():
    t = ClusterStatsTracker()
    t.step(PAD, PAD)  # all-padding -> no-op
    t.step(TOKENS, TIMES)
    n_clusters_per_sub, cluster_size = t.finalize()
    assert sorted(cluster_size.tolist()) == [1, 2]  # two events, sizes 2 and 1
    assert n_clusters_per_sub.tolist() == [1]  # row 0 has one multi-token cluster


def test_tied_event_cooccurrence():
    t = TiedEventTracker(vocab_size=8)
    t.step(PAD, PAD)  # all-padding -> no-op
    t.step(TOKENS, TIMES)
    heat = t.finalize()
    # only 5 & 7 co-occur (in row 0's event); diagonal is zeroed
    assert heat[5, 7] == 1 and heat[7, 5] == 1
    assert heat.sum() == 2  # nothing else


def test_cooccurrence_symmetric():
    t = CooccurrenceTracker(vocab_size=8)
    t.step(np.zeros(3, dtype=int), np.zeros(3, dtype=int))  # all-padding -> no-op
    t.step(SEQ_TOKENS, SEQ_TIMES)
    heat = t.finalize()
    # 5 & 7 co-occur across different days; the repeat of 5 counts once
    assert heat[5, 7] == 1 and heat[7, 5] == 1
    assert heat[5, 5] == 0  # diagonal zeroed; self-occurrence not counted
    assert heat.sum() == 2


def test_cooccurrence_before():
    t = CooccurrenceTracker(vocab_size=8, before=True)
    t.step(SEQ_TOKENS, SEQ_TIMES)
    heat = t.finalize()
    # 5 (first at day 10) is strictly before 7 (day 20); not the reverse
    assert heat[5, 7] == 1 and heat[7, 5] == 0
    assert heat.sum() == 1

    # same-day tokens are NOT strictly before one another -> nothing counted
    t2 = CooccurrenceTracker(vocab_size=8, before=True)
    t2.step(np.array([5, 7]), np.array([10, 10]))
    assert t2.finalize().sum() == 0


def test_missing_dx_tracker():
    # drug token 5 treats diseases {7, 8}; one participant per step.
    t = MissingDxTracker({5: {7, 8}})
    t.step(np.array([5, 7]))  # has dx 7 -> not a gap
    t.step(np.array([5, 1, 2]))  # drug 5, none of {7, 8} -> gap
    t.step(np.array([5, 8, 9]))  # has dx 8 -> not a gap
    t.step(np.array([1, 2, 3]))  # no drug 5 -> not counted
    assert t.finalize()[5] == (3, 1)  # 3 holders of drug 5, 1 with no managed dx


if __name__ == "__main__":
    test_flatten_events_skips_all_padding()
    test_flatten_events_drops_padding()
    test_cluster_stats()
    test_tied_event_cooccurrence()
    test_cooccurrence_symmetric()
    test_cooccurrence_before()
    test_missing_dx_tracker()
    print("ok")
