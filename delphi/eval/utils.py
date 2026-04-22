import numpy as np
import torch

from delphi.model.multimodal import nearest_input_pos
from delphi.multimodal import Modality


def corrective_indices(T0: torch.Tensor, T1: torch.Tensor, offset: float):
    assert T0.shape == T1.shape  # (m, n)
    T0_expanded = T0.unsqueeze(1)  # (m, 1, n)
    T1_expanded = T1.unsqueeze(-1)  # (m, n, 1)
    C = (T0_expanded < (T1_expanded - offset)).sum(dim=2) - 1

    return C.long()


def correct_time_offset(
    T0: torch.Tensor, T1: torch.Tensor, logits: torch.Tensor, offset: float
):
    """Shift T0/logits to approximate predictions at ``T1 - offset``.

    For each target j, gather ``T0``/``logits`` at the nearest input position
    strictly earlier than ``T1[j] - offset``. Targets with no such position get
    NaN (both T0 and logits), so downstream NaN-aware collators drop them.
    """
    corr_idx = nearest_input_pos(age=T0, targets_age=T1 - offset)
    invalid = corr_idx == -1
    corr_idx = torch.clamp(corr_idx, min=0)
    T0 = torch.gather(input=T0, index=corr_idx, dim=1)
    logits = torch.gather(
        input=logits,
        index=corr_idx.unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
        dim=1,
    )

    T0[invalid] = torch.nan
    logits[invalid] = torch.nan

    return T0, logits


def sample_boolean_mask(mask):
    """Sample one True value per row from a boolean mask (vectorized)."""
    n_rows = mask.shape[0]
    result = torch.zeros_like(mask).bool()

    # Count True values per row
    counts = mask.sum(dim=1)
    has_true = counts > 0

    if not has_true.any():
        return result

    # For rows with at least one True, generate random positions
    random_positions = torch.rand(n_rows, mask.shape[1])
    random_positions[~mask] = -torch.inf  # Mask out False positions

    # Select the position with max random value per row
    selected_cols = torch.argmax(random_positions, dim=1)
    result[torch.arange(n_rows), selected_cols] = has_true

    return result


class SexCollator:

    def __init__(self):
        self.is_female = list()

    def step(self, tokens):
        self.is_female.append((tokens == 2).any(dim=1).detach().cpu())

    def finalize(self):
        return torch.concat(self.is_female)


class EventTimeCollator:

    def __init__(self, vocab_size: int):
        self.exit_time = list()
        self.occur_time = list()
        self.vocab_size = vocab_size

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor):
        batch_size, _ = tokens.shape
        self.exit_time.append(timestep.detach().cpu().numpy().max(axis=1))

        occur_time = torch.full((batch_size, self.vocab_size), fill_value=torch.nan)
        occur_time = occur_time.scatter_(dim=1, index=tokens, src=timestep)
        self.occur_time.append(occur_time.detach().cpu().numpy())

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:

        occur_time = np.concatenate(self.occur_time, axis=0)
        exit_time = np.concatenate(self.exit_time, axis=0)

        return occur_time, exit_time


class LogitCollector:

    def __init__(self, age: float, n_repeats: int = 1):
        self.logits = list()
        self.n_repeats = n_repeats
        self.age = age

    def step(self, tokens: torch.Tensor, timestep: torch.Tensor, logits: torch.Tensor):

        batch_size, _, vocab_size = logits.shape
        logits[logits == -torch.inf] = torch.nan

        collect_idx = torch.argmax(torch.clamp(timestep, max=self.age), dim=1)
        collect_logits = logits[torch.arange(batch_size), collect_idx, :]
        collect_logits = torch.reshape(collect_logits, (-1, self.n_repeats, vocab_size))
        collect_logits = torch.nanmean(collect_logits, dim=1)

        self.logits.append(collect_logits.detach().cpu())

    def finalize(self):

        return torch.cat(self.logits, dim=0).numpy()


class ModalityCollator:

    def __init__(self, modalities: list[str]):
        self.modalities = [Modality[modality.upper()] for modality in modalities]
        self.max_mod = max([modality.value for modality in self.modalities])
        self.mod_timesteps = list()

    def step(self, mod_tokens, timesteps):
        assert mod_tokens.shape == timesteps.shape
        mod_timesteps = torch.full(
            (timesteps.shape[0], self.max_mod + 1), fill_value=torch.nan
        ).to(timesteps.device)
        mod_timesteps = mod_timesteps.scatter_(dim=1, src=timesteps, index=mod_tokens)
        self.mod_timesteps.append(mod_timesteps.detach().cpu())
        return mod_timesteps

    def finalize(self):
        return torch.cat(self.mod_timesteps, dim=0)
