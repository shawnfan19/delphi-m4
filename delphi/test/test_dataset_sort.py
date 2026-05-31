"""`MultimodalDataset.sort_by_length` reorders participants by sequence length.

Uses a stub reader with controllable per-participant lengths (no real data), so
the test stays light and pins down the contract c-index relies on: the method
returns a length-ordered permutation of the pids AND that returned array IS
`self.participants` (single source of truth for downstream per-participant arrays).
"""

import numpy as np

from delphi.data.dataset import MultimodalDataset

# pid -> raw token count; measured length is len(x0) = count - 1 (no biomarkers,
# no transforms), so ordering by measured length == ordering by these counts.
PID_TO_LEN = {10: 5, 20: 50, 30: 3, 40: 20, 50: 8}


class _StubReader:
    def __init__(self, pid_to_len):
        self.pid_to_len = pid_to_len
        self.tokenizer = {"pad": 0}

    def __getitem__(self, pid):
        n = self.pid_to_len[int(pid)]
        x = np.arange(n, dtype=np.int64)
        t = np.arange(n, dtype=np.float32)
        empty = np.array([], dtype=np.float32)
        return x, t, {}, empty, empty


def _dataset():
    pids = np.array(list(PID_TO_LEN), dtype=np.int64)
    return MultimodalDataset(reader=_StubReader(PID_TO_LEN), pids=pids)


def _measured_lengths(pids):
    return [PID_TO_LEN[int(p)] for p in pids]


def test_sort_by_length_ascending():
    ds = _dataset()
    original = set(ds.participants.tolist())

    out = ds.sort_by_length(progress=False)

    # returned array is the authoritative order (same object as self.participants)
    assert out is ds.participants
    # permutation: same set of pids, same count
    assert set(out.tolist()) == original
    assert len(ds) == len(PID_TO_LEN)
    # ascending by measured length
    lengths = _measured_lengths(out)
    assert lengths == sorted(lengths)
    assert out.tolist() == [30, 10, 50, 40, 20]


def test_sort_by_length_descending():
    ds = _dataset()
    out = ds.sort_by_length(descending=True, progress=False)

    assert out is ds.participants
    assert set(out.tolist()) == set(PID_TO_LEN)
    lengths = _measured_lengths(out)
    assert lengths == sorted(lengths, reverse=True)
