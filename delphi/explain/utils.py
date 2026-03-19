import torch

from delphi.multimodal import Modality


def pack_bio(
    bio_x_dict: dict[Modality, torch.Tensor], modalities: list[Modality]
) -> torch.Tensor:
    """Concatenate all biomarker tensors into a single flat vector."""
    parts = [bio_x_dict[mod].reshape(-1) for mod in modalities]
    return torch.cat(parts)


def unpack_bio(
    flat: torch.Tensor,
    bio_x_dict: dict[Modality, torch.Tensor],
    modalities: list[Modality],
) -> dict[Modality, torch.Tensor]:
    """Reconstruct bio_x_dict from a flat vector."""
    out = {}
    offset = 0
    for mod in modalities:
        n = bio_x_dict[mod].numel()
        out[mod] = flat[offset : offset + n].reshape(bio_x_dict[mod].shape)
        offset += n
    return out
