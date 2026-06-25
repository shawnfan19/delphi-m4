import numpy as np
from scipy import sparse


def _flatten_events(tokens: np.ndarray, timesteps: np.ndarray):
    """Flatten a padded ``(N, K)`` batch to its non-padding events.

    Drops padding (token 0), then returns, for each surviving token, its batch
    row index, token id, and timestep, plus the ``(row, time)`` column-stack
    used to group tokens into events (an event = a unique ``(row, time)`` pair).
    Returns ``None`` if the batch is all padding, so callers can skip it.
    """
    mask = tokens != 0
    if not np.any(mask):
        return None
    flat_tokens = tokens[mask]
    flat_times = timesteps[mask]
    N, K = tokens.shape
    flat_rows = np.arange(N).repeat(K).reshape(N, K)[mask]
    event_keys = np.column_stack((flat_rows, flat_times))
    return flat_rows, flat_tokens, flat_times, event_keys


class ClusterStatsTracker:
    def __init__(self):
        self.n_clusters_per_sub = list()
        self.cluster_size = list()

    def step(self, tokens: np.ndarray, timesteps: np.ndarray):
        flat = _flatten_events(tokens, timesteps)
        if flat is None:
            return  # Skip empty batches
        flat_rows, _, _, event_keys = flat

        # np.unique maps every (row, time) pair to a unique integer ID (0 to Num_Events-1)
        _, event_ids, cluster_size = np.unique(
            event_keys, axis=0, return_index=True, return_counts=True
        )

        event_subs = flat_rows[event_ids]
        cluster_subs = event_subs[cluster_size > 1]
        _, n_clusters_per_sub = np.unique(cluster_subs, return_counts=True)

        self.n_clusters_per_sub.append(n_clusters_per_sub)
        self.cluster_size.append(cluster_size)

    def finalize(self):
        return np.concatenate(self.n_clusters_per_sub), np.concatenate(
            self.cluster_size
        )


class _SparsePairTracker:
    """Shared V×V sparse co-occurrence accumulator.

    Subclasses implement ``step`` to add a batch/sequence's contribution; the
    only thing that differs between them is *what counts as a co-occurrence*.
    """

    def __init__(self, vocab_size):
        # vocab_size: V, the dimension of the vocabulary (max token id + 1).
        # CSR sparse running sum keeps accumulation cheap on memory.
        self.vocab_size = vocab_size
        self.global_cooccurrence = sparse.csr_matrix(
            (vocab_size, vocab_size), dtype=np.int32
        )

    def finalize(self, as_dense=True):
        """Zero the diagonal (drop self-occurrences) and return the V×V matrix.

        ``as_dense`` returns a numpy array; otherwise the sparse matrix.
        """
        result_matrix = self.global_cooccurrence.copy()  # don't corrupt running state
        result_matrix.setdiag(0)
        result_matrix.eliminate_zeros()
        return result_matrix.toarray() if as_dense else result_matrix


class TiedEventTracker(_SparsePairTracker):
    """Co-occurrence of tokens *tied to the same event* = same ``(participant, day)``.

    Two tokens co-occur only when they share an event key, so a finalized
    ``[a, b]`` counts the events (participant-days) in which both appear. Tokens
    for the same participant on different days do **not** co-occur.

    ``step`` takes batched ``tokens``/``timesteps`` arrays, each ``(N, K)``.
    """

    def step(self, tokens, timesteps):
        flat = _flatten_events(tokens, timesteps)
        if flat is None:
            return  # Skip empty batches
        _, flat_tokens, _, event_keys = flat

        # np.unique maps every (row, time) pair to a unique event ID.
        _, event_ids = np.unique(event_keys, axis=0, return_inverse=True)
        num_events = event_ids.max() + 1
        # Incidence matrix (events × vocab); a token repeated in one event sums up.
        ones = np.ones(len(flat_tokens), dtype=np.int32)
        X_batch = sparse.csr_matrix(
            (ones, (event_ids, flat_tokens)), shape=(num_events, self.vocab_size)
        )
        # (V × events) @ (events × V) -> (V × V)
        self.global_cooccurrence += X_batch.T @ X_batch


class CooccurrenceTracker(_SparsePairTracker):
    """Participant-level (lifetime) token co-occurrence — one participant per ``step``.

    Two tokens co-occur when they both appear *anywhere* in the participant's
    sequence, regardless of when. Each token counts once (binary presence at its
    first occurrence), so a finalized ``[i, j]`` counts the participants in whom
    both appear.

    With ``before=True`` the matrix is *directed*: ``[i, j]`` is incremented only
    when ``i``'s first occurrence is *strictly earlier* than ``j``'s, so same-day
    pairs are dropped. With ``before=False`` (default) it is symmetric and
    same-day pairs count.

    ``step`` takes a single participant's 1-D ``tokens`` (token 0 = padding /
    masked-out, dropped). ``timesteps`` (1-D, same length) is required when
    ``before=True`` and ignored otherwise.
    """

    # Merging a (V×V) matrix into the running sum costs O(nnz) every time, so we
    # buffer per-participant (src, dst) pairs and merge in bulk; coo_matrix sums
    # duplicates on conversion, giving the same result far faster.
    _FLUSH_AT = 5_000_000  # buffered pairs before a merge

    def __init__(self, vocab_size, before=False):
        super().__init__(vocab_size)
        self.before = before
        self._src, self._dst, self._buffered = [], [], 0

    def step(self, tokens, timesteps=None):
        keep = tokens != 0
        toks = tokens[keep]
        if toks.size == 0:
            return  # Skip empty sequences

        if not self.before:
            u = np.unique(toks)  # binary presence; repeats collapse
            src, dst = np.repeat(u, len(u)), np.tile(
                u, len(u)
            )  # all pairs; diag dropped in finalize
        else:
            if timesteps is None:
                raise ValueError("before=True requires timesteps")
            times = timesteps[keep]
            order = np.argsort(times, kind="stable")  # earliest first
            u, first = np.unique(toks[order], return_index=True)
            ut = times[order][first]  # first-occurrence time of each distinct token
            ii, jj = np.nonzero(ut[:, None] < ut[None, :])  # i strictly before j
            src, dst = u[ii], u[jj]

        self._src.append(src)
        self._dst.append(dst)
        self._buffered += len(src)
        if self._buffered >= self._FLUSH_AT:
            self._flush()

    def _flush(self):
        if not self._src:
            return
        src, dst = np.concatenate(self._src), np.concatenate(self._dst)
        ones = np.ones(len(src), dtype=np.int32)
        self.global_cooccurrence += sparse.coo_matrix(
            (ones, (src, dst)), shape=(self.vocab_size, self.vocab_size)
        ).tocsr()
        self._src, self._dst, self._buffered = [], [], 0

    def finalize(self, as_dense=True):
        self._flush()
        return super().finalize(as_dense)
