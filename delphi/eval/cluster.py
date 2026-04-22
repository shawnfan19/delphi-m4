import numpy as np
from scipy import sparse


class ClusterStatsTracker:
    def __init__(self):
        self.n_clusters_per_sub = list()
        self.cluster_size = list()

    def step(self, tokens: np.ndarray, timesteps: np.ndarray):

        # 1. Mask Padding
        # Filter out 0 (padding) tokens
        mask = tokens != 0
        if not np.any(mask):
            return  # Skip empty batches
        flat_tokens = tokens[mask]
        flat_times = timesteps[mask]

        # Get row indices (0 to N-1) for every valid token
        N, K = tokens.shape
        row_indices = np.arange(N).repeat(K).reshape(N, K)
        flat_rows = row_indices[mask]
        # 2. Identify Unique Events
        # An event is a unique combination of (Batch_Row_Index, Timestep)
        # We stack them to create unique keys for grouping
        event_keys = np.column_stack((flat_rows, flat_times))

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
        # 1. Mask Padding
        # Filter out 0 (padding) tokens
        mask = tokens != 0
        if not np.any(mask):
            return  # Skip empty batches
        flat_tokens = tokens[mask]
        flat_times = timesteps[mask]

        # Get row indices (0 to N-1) for every valid token
        N, K = tokens.shape
        row_indices = np.arange(N).repeat(K).reshape(N, K)
        flat_rows = row_indices[mask]
        # 2. Identify Unique Events
        # An event is a unique combination of (Batch_Row_Index, Timestep)
        # We stack them to create unique keys for grouping
        event_keys = np.column_stack((flat_rows, flat_times))

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
