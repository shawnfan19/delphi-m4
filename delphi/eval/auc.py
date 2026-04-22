import numpy as np
import torch
from scipy.stats import rankdata

from delphi.eval.utils import sample_boolean_mask


def mann_whitney_auc(x1: np.ndarray, x2: np.ndarray) -> float:

    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    n1 = len(x1)
    n2 = len(x2)
    x12 = np.concatenate([x1, x2])
    ranks = rankdata(x12, method="average")

    R1 = ranks[:n1].sum()
    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    if n1 == 0 or n2 == 0:
        return np.nan
    return U1 / n1 / n2


def batched_mann_whitney_auc(
    scores: np.ndarray, ctl: np.ndarray, case: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise AUC over (N, V) score matrix.

    Scores outside `ctl | case` or NaN are excluded from ranking (per column).
    Returns (ctl_counts, case_counts, auc), each of shape (V,).
    """
    assert scores.shape == ctl.shape == case.shape
    masked = np.where(ctl | case, scores, np.nan)
    ranks = rankdata(masked, method="average", axis=0, nan_policy="omit")

    valid = ~np.isnan(masked)
    n1 = (ctl & valid).sum(axis=0)
    n2 = (case & valid).sum(axis=0)
    R1 = np.where(ctl & valid, ranks, 0).sum(axis=0)

    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    denom = n1 * n2
    auc = np.full(denom.shape, np.nan, dtype=float)
    np.divide(U1, denom, out=auc, where=denom > 0)
    return n1, n2, auc


class AgeStratRatesCollator:

    def __init__(self, age_groups: torch.Tensor):
        self.age_groups = age_groups
        self.ctl_rates = list()
        self.ctl_times = list()

    def step(
        self,
        timesteps: torch.Tensor,
        logits: torch.Tensor,
    ):

        batch_size = logits.shape[0]
        n_age_bins = len(self.age_groups) - 1
        bin_assignments = torch.searchsorted(self.age_groups, timesteps, right=True)
        bin_assignments -= 1

        ctl_rates = list()
        ctl_times = list()
        for bin_idx in range(n_age_bins):
            bin_mask = sample_boolean_mask(bin_assignments == bin_idx)
            ctl_rate = torch.full(
                (batch_size, logits.shape[-1]),
                dtype=logits.dtype,
                fill_value=torch.nan,
            ).to(logits.device)
            ctl_time = torch.full(
                (batch_size,), dtype=timesteps.dtype, fill_value=torch.nan
            ).to(logits.device)
            ctl_rate[bin_mask.any(dim=-1)] = logits[bin_mask, :]
            ctl_time[bin_mask.any(dim=-1)] = timesteps[bin_mask]
            ctl_rates.append(ctl_rate)
            ctl_times.append(ctl_time)
        ctl_rates = torch.stack(ctl_rates, dim=1)
        ctl_times = torch.stack(ctl_times, dim=1)

        self.ctl_rates.append(ctl_rates.detach().cpu())
        self.ctl_times.append(ctl_times.detach().cpu())

    def finalize(self):
        return torch.concat(self.ctl_rates), torch.concat(self.ctl_times)


class DiseaseRatesCollator:

    def __init__(self, targets: torch.Tensor):
        self.targets = targets
        self.dis_rates = list()
        self.dis_times = list()

    def step(
        self,
        tokens: torch.Tensor,
        timesteps: torch.Tensor,
        logits: torch.Tensor,
    ):

        dis_time = torch.full(
            (logits.shape[0], logits.shape[-1]),
            dtype=timesteps.dtype,
            fill_value=torch.nan,
        ).to(logits.device)
        dis_time.scatter_(index=tokens, src=timesteps, dim=1)
        self.dis_times.append(dis_time.detach().cpu())

        dis_rate = torch.full(
            (logits.shape[0], logits.shape[-1]),
            dtype=logits.dtype,
            fill_value=torch.nan,
        ).to(logits.device)
        uniq_tokens = torch.unique(tokens)
        uniq_tokens = uniq_tokens[torch.isin(uniq_tokens, self.targets)]
        for token in uniq_tokens:
            have_disease = tokens == token
            dis_rate[have_disease.any(dim=1), token] = logits[have_disease][:, token]
        self.dis_rates.append(dis_rate.detach().cpu())

    def finalize(self):
        return torch.concat(self.dis_rates), torch.concat(self.dis_times)


class ConcordanceCollator:

    def __init__(
        self,
        dis_rates,
        onset_times,
        is_female,
        offset,
        chunk_size=8192,
        max_gap_days=1826.25,
        cutoff=None,
    ):
        # Flatten case events: each non-NaN entry in dis_rates is a case
        case_participants, case_tokens = (~torch.isnan(dis_rates)).nonzero(
            as_tuple=True
        )
        self.case_scores = dis_rates[case_participants, case_tokens].float()
        self.case_times = onset_times[case_participants, case_tokens].float()
        self.case_tokens = case_tokens
        self.case_participants = case_participants
        self.case_sex = is_female[case_participants].cpu().numpy()

        self.query_times = self.case_times - offset
        self.onset_times = onset_times
        self.chunk_size = chunk_size
        self.max_gap_days = max_gap_days
        self.cutoff = cutoff

        E = len(case_participants)
        self.concordant_pairs = np.zeros(E, dtype=np.float64)
        self.total_pairs = np.zeros(E, dtype=np.float64)
        self.participant_offset = 0

    def step(self, age, scores):
        B, L, V = scores.shape
        device = scores.device
        E_total = len(self.case_tokens)
        j_globals = torch.arange(B, device=device) + self.participant_offset

        for e_start in range(0, E_total, self.chunk_size):
            e_end = min(e_start + self.chunk_size, E_total)
            E_c = e_end - e_start

            chunk_query_times = self.query_times[e_start:e_end]
            chunk_case_times = self.case_times[e_start:e_end]
            chunk_tokens = self.case_tokens[e_start:e_end]
            chunk_participants = self.case_participants[e_start:e_end]
            chunk_scores = self.case_scores[e_start:e_end]

            # Batched searchsorted: (B, L) sorted × (B, E_c) queries → (B, E_c) indices
            idx_mat = (
                torch.searchsorted(
                    age.contiguous(),
                    chunk_query_times.unsqueeze(0).expand(B, -1).contiguous(),
                    right=True,
                )
                - 1
            )
            idx_c = idx_mat.clamp(0, L - 1)

            # Timestamps and scores at each found position
            t_at = age.gather(1, idx_c)
            flat_b = (
                torch.arange(B, device=device).unsqueeze(1).expand(-1, E_c).reshape(-1)
            )
            ctrl_scores = scores[
                flat_b,
                idx_c.reshape(-1),
                chunk_tokens.unsqueeze(0).expand(B, -1).reshape(-1),
            ].reshape(B, E_c)

            # Validity: within timeline and not padding
            valid = (idx_mat >= 0) & (t_at > 0)
            # Max gap: control score must be within max_gap of query time
            valid &= (chunk_query_times.unsqueeze(0) - t_at) < self.max_gap_days
            # Control score must be after control's biomarker cutoff
            if self.cutoff is not None:
                valid &= t_at >= self.cutoff[j_globals].unsqueeze(1)
            # At-risk: control had not yet developed disease at the case's event time
            j_onset = self.onset_times[
                j_globals.unsqueeze(1), chunk_tokens.unsqueeze(0).expand(B, -1)
            ]
            valid &= j_onset.isnan() | (j_onset > chunk_case_times.unsqueeze(0))
            # Do not compare a case to itself
            valid &= j_globals.unsqueeze(1) != chunk_participants.unsqueeze(0)

            self.concordant_pairs[e_start:e_end] += (
                (valid & (ctrl_scores.float() < chunk_scores.unsqueeze(0)))
                .sum(0)
                .cpu()
                .numpy()
            )
            self.total_pairs[e_start:e_end] += valid.sum(0).cpu().numpy()

        self.participant_offset += B

    def finalize(self):
        return (
            self.case_sex,
            self.case_tokens.cpu().numpy(),
            self.total_pairs,
            self.concordant_pairs,
        )
