from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cloudpathlib import AnyPath

from delphi.data import MultimodalDataset
from delphi.data.transform import TokenTransform
from delphi.data.ukb import MultimodalUKBReader
from delphi.env import DELPHI_CKPT_READ
from delphi.experiment import CliConfig, load_ckpt, move_batch_to_device
from delphi.model.tpp import tpp_dispatch


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    ckpt: str
    event: str
    n_samples: int = 1
    steps_per_interval: int = 10


args = TaskConfig.from_cli()
args.print()

ckpt = AnyPath(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

val_pids = MultimodalUKBReader.participants("val")
reader = MultimodalUKBReader(biomarkers=None)
val_pids = reader.participants_with_event(pids=val_pids, event=args.event)
rng = np.random.default_rng(seed=42)
val_pids = rng.permutation(val_pids)
if args.n_samples < len(val_pids):
    val_pids = val_pids[: args.n_samples]

token_transform_args = ckpt_dict["token_transform_args"]
token_transform = TokenTransform(**token_transform_args)
token_transform.describe()

ds = MultimodalDataset(
    reader=reader,
    pids=val_pids,
    token_transform=token_transform,
)

for i in range(len(ds)):
    x0, t0, _, _, _, x1, t1 = move_batch_to_device(ds.get_batch([i]), device=device)
    outputs, _, _ = model(x0, t0)
    tpp = tpp_dispatch(model, outputs)
    # Reshape to (1, 1, N) for the interpolate function
    t1 = t1.unsqueeze(0)
    # Calculate the new total length
    # N elements means N-1 intervals.
    new_length = (t1.numel() - 1) * args.steps_per_interval + 1
    # Interpolate (align_corners=True ensures it perfectly hits the original numbers)
    t_grid = F.interpolate(t1, size=new_length, mode="linear", align_corners=True)
    t_grid = t_grid.squeeze(0)

    intensity, _ = tpp.intensity(t_grid)
    event_intensity = intensity[..., reader.tokenizer[args.event]]

    t_grid = t_grid.detach().cpu().numpy().ravel()
    event_intensity = event_intensity.detach().cpu().numpy().ravel()
    plt.figure()
    plt.plot(t_grid, event_intensity)
    plt.yscale("log")
    plt.ylabel("intensity")
    plt.xlabel("time")
    plt.title(args.event)
    plt.show()
