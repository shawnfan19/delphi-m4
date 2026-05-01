import numpy as np
import torch


class PopulationKaplanMeierEstimator:

    def __init__(self, timestep: np.ndarray, tokens: np.ndarray, vocab_size: int):

        assert timestep.shape == tokens.shape
        n_subjects = tokens.shape[0]

        # surv_time[i, j] -> exit time for event i in subject j
        # note that exit time can be
        # – time of event (for first occurrence data)
        # – time of death
        # – time at last follow-up
        surv_time = timestep.max(axis=1)[:, None]
        surv_time = np.repeat(surv_time, vocab_size, axis=1)
        np.put_along_axis(arr=surv_time, indices=tokens, values=timestep, axis=1)
        surv_time = surv_time.transpose(1, 0)

        # occur[i, j] -> 1 if subject j experiences event i else 0
        occur = np.zeros((n_subjects, 1))
        occur = np.repeat(occur, vocab_size, axis=1)
        np.put_along_axis(arr=occur, indices=tokens, values=1, axis=1)
        occur = occur.transpose(1, 0)

        sort_surv_time = np.argsort(surv_time, axis=1)
        surv_time = np.take_along_axis(surv_time, indices=sort_surv_time, axis=1)
        occur = np.take_along_axis(occur, indices=sort_surv_time, axis=1)

        surv_percent = list()
        surv_timestep = list()
        for i in range(vocab_size):
            uniq_time, inverse_indices, n_exit = np.unique(
                surv_time[i, :], return_inverse=True, return_counts=True
            )
            n_exit = np.concatenate(([0], n_exit[:-1]))
            n_occur = np.bincount(inverse_indices, weights=occur[i, :])
            n_surv = n_subjects - np.cumsum(n_exit)
            surv_percent.append(np.cumprod(1 - n_occur / n_surv))
            surv_timestep.append(uniq_time)

        self.surv_percent = surv_percent
        self.surv_time = surv_timestep

    def incidence(self, start_age: float, end_age: float) -> np.ndarray:

        incidence = list()
        for token in range(len(self.surv_percent)):
            start_mask = self.surv_time[token] <= start_age
            end_mask = self.surv_time[token] <= end_age
            if (start_mask.sum() == 0) or (end_mask.sum() == 0):
                incidence.append(np.nan)
            else:
                start_surv = self.surv_percent[token][start_mask].min()
                end_surv = self.surv_percent[token][end_mask].min()
                incidence.append((start_surv - end_surv) / start_surv)
        return np.array(incidence)


class KaplanMeierEstimator:

    def __init__(self, surv_timesteps: np.ndarray, occur: np.ndarray):

        n_subjects = surv_timesteps.size
        uniq_time, inverse_indices, n_exit = np.unique(
            surv_timesteps, return_inverse=True, return_counts=True
        )
        n_exit = np.concatenate(([0], n_exit[:-1]))
        n_occur = np.bincount(inverse_indices, weights=occur)
        n_surv = n_subjects - np.cumsum(n_exit)
        self.surv_percent = np.cumprod(1 - n_occur / n_surv)
        self.surv_time = uniq_time

    def incidence(self, start_age, end_age):
        start_idx = np.searchsorted(self.surv_time, start_age, side="right") - 1
        end_idx = np.searchsorted(self.surv_time, end_age, side="right") - 1
        start_surv = 1.0 if start_idx < 0 else self.surv_percent[start_idx]
        end_surv = 1.0 if end_idx < 0 else self.surv_percent[end_idx]
        return (start_surv - end_surv) / start_surv


class NelsonAalenEstimator:

    def __init__(
        self,
        timesteps: torch.Tensor,
        intensities: torch.Tensor,
        at_risk: torch.Tensor,
    ):
        """
        args:
            timesteps (B, L)
            intensities (B, L, V)
            at_risk (B, L, V)
        """

        assert timesteps.shape[1] == intensities.shape[1] == at_risk.shape[1]
        assert (intensities[~at_risk.bool()] == 0).all()

        self.timesteps = torch.unique(timesteps[timesteps >= 0])
        self.intervals = self.timesteps.diff()

        idx = (
            torch.searchsorted(
                timesteps,
                self.timesteps.unsqueeze(0).expand(timesteps.shape[0], -1),
                right=True,
            )
            - 1
        )
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, intensities.shape[-1])
        intensities = torch.gather(intensities, 1, idx_expanded)
        at_risk = torch.gather(at_risk, 1, idx_expanded)

        denom = at_risk.sum(dim=0)
        self.hazard_rate = torch.where(
            denom > 0, intensities.sum(dim=0) / denom, other=0
        )
        areas = self.hazard_rate[:-1] * self.intervals.unsqueeze(-1)
        self.cumul_hazard = torch.zeros_like(self.hazard_rate)
        self.cumul_hazard[1:] = torch.cumsum(areas, dim=0)

        assert self.timesteps.shape[0] == self.cumul_hazard.shape[0]

    def __call__(self, t: float | torch.Tensor):

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.timesteps.device)
        idx = torch.searchsorted(self.timesteps, t, right=True) - 1
        # assert (idx >= 0).all(), "queried time is before the first timestep"

        time_in_interval = t - self.timesteps[idx]
        hazard = self.cumul_hazard[idx] + (
            time_in_interval.unsqueeze(-1) * self.hazard_rate[idx]
        )

        return hazard


def kaplan_meier_incidence(
    surv_prob: np.ndarray, surv_time: np.ndarray, start: float, end: float
):
    """
    assumes the same survival times across all diseases and all participants

    ! this can be vectorized but is actually faster AS IS when # participants is large

    inputs:
        – surv_prob [# participants, # tokens, # time_intervals]
        – surv_time [# time_intervals]
    output(s):
        – incidence [# participants, # tokens]
    """
    assert len(surv_prob.shape) == 3
    assert len(surv_time.shape) == 1
    assert surv_prob.shape[-1] == surv_time.size

    start_mask = surv_time <= start
    end_mask = surv_time <= end

    incidence = list()
    for token in range(surv_prob.shape[1]):
        start_surv = surv_prob[:, token, start_mask].min(axis=-1)
        end_surv = surv_prob[:, token, end_mask].min(axis=-1)
        incidence.append((start_surv - end_surv) / start_surv)

    return np.stack(incidence, axis=1)


def integrate_risk(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    timesteps: torch.Tensor,
    time_intervals: torch.Tensor,
):
    """
    not vectorized due to memory concerns when time_intervals are dense

    input(s):
        - hazard_rates: [# participants, # timesteps, # tokens]
        - timesteps: [# participants, # timesteps]
        - time_intervals: [# intervals]
        - last_time_by_event: [# participants, # tokens]
    output(s):
        - risk: [# participants, # tokens, # intervals]
    """
    _, _, vocab_size = logits.shape

    logits[logits == -torch.inf] = torch.nan
    hazard_rates = logits[:, :-1].exp()

    last_time_by_event = (
        timesteps.max(dim=1, keepdim=True)[0].expand(-1, vocab_size).clone()
    )
    last_time_by_event = last_time_by_event.scatter_(index=tokens, src=timesteps, dim=1)

    starts = time_intervals[:-1]
    ends = time_intervals[1:]
    risks = list()
    for start, end in zip(starts, ends):
        _timestep = timesteps.unsqueeze(-1)
        _timestep = torch.clamp(_timestep, min=start)
        _timestep = torch.clamp(_timestep, max=last_time_by_event.unsqueeze(1))
        _timestep = torch.clamp(_timestep, max=end)
        # _timestep: [# participants, # timesteps, # tokens]
        delta_t = torch.diff(_timestep, dim=1)
        not_enough_exposure = torch.nansum(delta_t, dim=1) < (end - start)

        cumul_hazard = delta_t * hazard_rates
        all_nan = torch.isnan(cumul_hazard).all(dim=1)
        cumul_hazard = torch.nansum(cumul_hazard, dim=1)
        # cumul_hazard: [# participants, # tokens]
        # manually set sum of NaNs to Nan because torch.nansum over all NaNs returns 0
        cumul_hazard[all_nan] = torch.nan
        cumul_hazard[not_enough_exposure] = torch.nan

        risk = 1 - torch.exp(-cumul_hazard)
        risks.append(risk)

    return torch.stack(risks, dim=-1)


class OnlineSurvivalEstimator:

    def __init__(self, time_intervals: np.ndarray, vocab_size: int, n_repeats: int = 1):
        self.time_intervals = time_intervals
        self.n_intervals = len(time_intervals) - 1
        self.risk_per_interval = np.zeros((vocab_size, self.n_intervals))
        self.counter = self.risk_per_interval.copy()
        self.n_repeats = n_repeats

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor, logits: torch.Tensor):

        _, _, vocab_size = logits.shape

        risk_per_interval = integrate_risk(
            logits=logits,
            tokens=tokens,
            timesteps=timestep,
            time_intervals=torch.tensor(self.time_intervals).to(tokens.device),
        )
        risk_per_interval = torch.reshape(
            risk_per_interval,
            (-1, self.n_repeats, vocab_size, len(self.time_intervals) - 1),
        )  # participants, # repeats, # vocab_size, # time_intervals
        risk_per_interval = torch.nanmean(risk_per_interval, dim=1)
        # participants, # vocab_size, # time_intervals
        risk_per_interval = risk_per_interval.detach().cpu().numpy()

        self.risk_per_interval += np.nansum(risk_per_interval, axis=0)
        self.counter += (~np.isnan(risk_per_interval)).sum(axis=0)

    def finalize(self):
        self.risk_per_interval /= self.counter
        return np.cumprod(1 - self.risk_per_interval, axis=-1), self.time_intervals


class IntervalKaplanMeierCollator:

    def __init__(
        self,
        time_horizon: list[float],
        start_age: float,
        time_intervals: None | list[float] = None,
        n_repeats: int = 1,
    ):
        self.time_intervals = time_intervals
        self.time_horizon = time_horizon
        self.start_age = start_age
        self.n_repeats = n_repeats
        self.prob_by_horizon = defaultdict(list)

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor, logits: torch.Tensor):

        if self.time_intervals is None:
            time_intervals = (
                torch.unique(torch.clamp(timestep, min=0), sorted=True)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            time_intervals = self.time_intervals

        risk_per_interval = integrate_risk(
            logits=logits,
            tokens=tokens,
            timesteps=timestep,
            time_intervals=torch.tensor(self.time_intervals).to(tokens.device),
        )
        risk_per_interval = torch.reshape(
            risk_per_interval,
            (-1, self.n_repeats, logits.shape[-1], len(time_intervals) - 1),
        )  # participants, # repeats, # vocab_size, # time_intervals
        risk_per_interval = torch.nanmean(risk_per_interval, dim=1)
        # participants, # vocab_size, # time_intervals

        surv_prob = torch.cumprod(1 - risk_per_interval, dim=-1)
        surv_time = np.array(time_intervals)[1:]
        for horizon in self.time_horizon:
            self.prob_by_horizon[horizon].append(
                kaplan_meier_incidence(
                    surv_prob=surv_prob.detach().cpu().numpy(),
                    surv_time=surv_time,
                    start=self.start_age,
                    end=self.start_age + horizon,
                )
            )

    def finalize(self):
        prob_by_horizon = dict()
        for horizon in self.prob_by_horizon.keys():
            prob_by_horizon[horizon] = np.concatenate(
                self.prob_by_horizon[horizon], axis=0
            )
        return prob_by_horizon


class SamplingProbCollator:

    def __init__(
        self, vocab_size: int, time_horizon: list, start_age: float, n_repeats: int = 1
    ):
        self.vocab_size = vocab_size
        self.time_horizon = time_horizon
        self.start_age = start_age
        self.n_repeats = n_repeats
        self.prob_by_horizon = defaultdict(list)

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor):

        batch_size, _ = tokens.shape

        occur_time = torch.full(
            (batch_size, self.vocab_size), fill_value=torch.nan, device=tokens.device
        )
        occur_time = (
            occur_time.scatter_(dim=1, index=tokens, src=timestep)
            .detach()
            .cpu()
            .numpy()
        )
        exit_time = timestep.detach().cpu().numpy().max(axis=1)

        for horizon in self.time_horizon:
            end_age = self.start_age + horizon
            occur = np.zeros((batch_size, self.vocab_size))
            occur[np.logical_and(occur_time > self.start_age, occur_time < end_age)] = 1
            occur[occur_time <= self.start_age] = float("nan")
            early_exit = exit_time < end_age
            early_exit = early_exit[:, None]
            occur[np.logical_and(early_exit, occur == 0)] = float("nan")
            occur = np.reshape(occur, (-1, self.n_repeats, occur.shape[-1]))
            occur = np.nanmean(occur, axis=1)
            self.prob_by_horizon[horizon].append(occur)

    def finalize(self):
        prob_by_horizon = dict()
        for horizon in self.prob_by_horizon.keys():
            prob_by_horizon[horizon] = np.concatenate(
                self.prob_by_horizon[horizon], axis=0
            )
        return prob_by_horizon
