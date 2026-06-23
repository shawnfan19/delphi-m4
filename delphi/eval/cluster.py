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


class CooccurrenceTracker:
    def __init__(self, vocab_size):
        """
        Initializes the tracker.

        Parameters:
        - vocab_size: int, the dimension V of the vocabulary (max token id + 1).
        """
        self.vocab_size = vocab_size

        # We use a sparse matrix for the running sum to save memory during accumulation.
        # CSR format is efficient for arithmetic operations.
        self.global_cooccurrence = sparse.csr_matrix(
            (vocab_size, vocab_size), dtype=np.int32
        )

    def step(self, tokens, timesteps):
        """
        Updates the co-occurrence counts with a new batch of data.

        Parameters:
        - tokens: (N, K) numpy array of integers.
        - timesteps: (N, K) numpy array of discrete days.
        """
        flat = _flatten_events(tokens, timesteps)
        if flat is None:
            return  # Skip empty batches
        _, flat_tokens, _, event_keys = flat

        # np.unique maps every (row, time) pair to a unique integer ID (0 to Num_Events-1)
        _, event_ids = np.unique(event_keys, axis=0, return_inverse=True)
        num_events = event_ids.max() + 1
        # 3. Create Incidence Matrix for this Batch (Events x Vocab)
        # Rows = Events, Cols = Token IDs
        # Values = 1 (presence).
        # Note: If a token appears twice in one event, the values sum up.
        ones = np.ones(len(flat_tokens), dtype=int)

        X_batch = sparse.csr_matrix(
            (ones, (event_ids, flat_tokens)), shape=(num_events, self.vocab_size)
        )
        # 4. Compute Batch Co-occurrence via Dot Product
        # (V x Events) @ (Events x V) -> (V x V)
        batch_cooccurrence = X_batch.T @ X_batch
        # 5. Update Global State
        self.global_cooccurrence += batch_cooccurrence

    def finalize(self, as_dense=True):
        """
        Finalizes the calculation, removes self-occurrences, and returns the heatmap.

        Parameters:
        - as_dense: bool. If True, returns a numpy array. If False, returns sparse matrix.

        Returns:
        - heatmap: (V, V) matrix.
        """
        # Work on a copy to avoid corrupting the running state if called multiple times
        result_matrix = self.global_cooccurrence.copy()

        # The prompt asks for co-occurrence with "any OTHER token".
        # We set the diagonal to 0 to remove self-occurrences (Token A with Token A).
        result_matrix.setdiag(0)

        # Eliminate any zeros created by setdiag from the sparse structure
        result_matrix.eliminate_zeros()

        if as_dense:
            return result_matrix.toarray()
        else:
            return result_matrix
