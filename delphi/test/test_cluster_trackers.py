"""ClusterStatsTracker / CooccurrenceTracker over a tiny hand-checked batch.

Pins the shared ``_flatten_events`` extraction: a ``(row, time)`` pair groups
tokens into one event, all-padding batches are skipped, and the two trackers'
outputs match the by-hand expectation.
"""

import numpy as np

from delphi.eval.cluster import (
    ClusterStatsTracker,
    CooccurrenceTracker,
    _flatten_events,
)

# row 0: tokens 5 & 7 at day 10 (a cluster of 2); row 1: token 5 at day 10.
TOKENS = np.array([[5, 7, 0], [5, 0, 0]])
TIMES = np.array([[10, 10, 0], [10, 0, 0]])
PAD = np.zeros((2, 3), dtype=int)


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


def test_cooccurrence():
    t = CooccurrenceTracker(vocab_size=8)
    t.step(PAD, PAD)  # all-padding -> no-op
    t.step(TOKENS, TIMES)
    heat = t.finalize()
    # only 5 & 7 co-occur (in row 0's event); diagonal is zeroed
    assert heat[5, 7] == 1 and heat[7, 5] == 1
    assert heat.sum() == 2  # nothing else


if __name__ == "__main__":
    test_flatten_events_skips_all_padding()
    test_flatten_events_drops_padding()
    test_cluster_stats()
    test_cooccurrence()
    print("ok")
