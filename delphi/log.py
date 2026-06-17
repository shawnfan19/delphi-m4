import os
from datetime import datetime
from pprint import pprint

import torch
import wandb
from cloudpathlib import AnyPath

from delphi import distributed
from delphi.env import DELPHI_CKPT_DIR


def _compress_int_runs(lst: list) -> list:
    if not lst or not all(type(x) is int for x in lst):
        return lst
    runs = []
    start = prev = lst[0]
    for x in lst[1:]:
        if x == prev + 1:
            prev = x
            continue
        runs.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = x
    runs.append(str(start) if start == prev else f"{start}-{prev}")
    return runs


def _format_for_display(obj):
    if isinstance(obj, dict):
        return {k: _format_for_display(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return _compress_int_runs(obj)
    return obj


class Logger:
    """Receives metric dicts and sends them to wandb / stdout."""

    def __init__(
        self,
        config: dict,
        backend: distributed.backend.DistributedBackend,
        wandb_log: bool = True,
        wandb_project: str = "delphi",
        run_name: None | str = None,
        summary: None | dict = None,
    ):
        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.run_name = run_name
        self.backend = backend
        self.wandb = wandb_log

        if backend.is_master_process():
            print("=== config ===")
            pprint(_format_for_display(config), indent=2, width=60)

            if self.wandb:
                import re

                wandb_project = re.sub(r"[^a-zA-Z0-9_\-.]", "_", wandb_project)
                wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=config,
                )
                wandb.define_metric("step")
                wandb.define_metric("lr", step_metric="step")
                wandb.define_metric("val/*", step_metric="step")
                wandb.define_metric("train/*", step_metric="step")
                wandb.define_metric("grad_norm/*", step_metric="step")
                wandb.define_metric("output/*", step_metric="step")
                wandb.define_metric("param/*", step_metric="step")

                if summary is not None:
                    for k, v in summary.items():
                        wandb.summary[k] = v

    def log(self, metrics: dict):
        if self.backend.is_master_process():
            if self.wandb:
                wandb.log(metrics)

    def log_grad_norm(self, model: torch.nn.Module):
        if self.backend.is_master_process():
            if self.wandb:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log(
                            {f"grad_norm/{name}": param.grad.norm().item()},
                            commit=False,
                        )

    def log_param_stats(self, model: torch.nn.Module):
        if self.backend.is_master_process():
            if self.wandb:
                for name, param in model.named_parameters():
                    tensor = param.detach().float()
                    wandb.log(
                        {
                            f"param/{name}/mean": tensor.mean().item(),
                            f"param/{name}/median": tensor.median().item(),
                            f"param/{name}/max": tensor.max().item(),
                            f"param/{name}/min": tensor.min().item(),
                        },
                        commit=False,
                    )

    def log_output(self, output: dict[str, torch.Tensor]):
        if self.backend.is_master_process():
            if self.wandb:
                for name, tensor in output.items():
                    if isinstance(tensor, dict):
                        continue
                    if not torch.is_floating_point(tensor):
                        continue
                    tensor = tensor.detach().float()
                    wandb.log(
                        {
                            f"output/{name}/mean": tensor.mean().item(),
                            f"output/{name}/median": tensor.median().item(),
                            f"output/{name}/max": tensor.max().item(),
                            f"output/{name}/min": tensor.min().item(),
                        },
                        commit=False,
                    )

    def print(self, msg: str):
        if self.backend.is_master_process():
            print(msg)


class Checkpointer:
    """Saves and loads training checkpoints."""

    def __init__(
        self,
        dump_dir: os.PathLike,
        backend: distributed.backend.DistributedBackend,
        metadata: dict | None = None,
    ):
        # AnyPath -> a local pathlib.Path or a cloudpathlib CloudPath, so a
        # gs:// DELPHI_CKPT_DIR writes checkpoints straight to GCS (symmetric
        # with load_ckpt). cloudpathlib uploads on close, atomically.
        self.dump_dir = AnyPath(DELPHI_CKPT_DIR) / str(dump_dir)
        self.backend = backend
        self.metadata = metadata or {}

        if backend.is_master_process():
            self.dump_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        step: int,
        best_val_loss: float = float("inf"),
        ckpt_fname: str = "ckpt.pt",
    ):
        if self.backend.is_master_process():
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            checkpoint = {
                "model": model.state_dict(),
                "model_type": model.model_type,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "iter_num": step,
                "best_val_loss": best_val_loss,
            } | self.metadata
            ckpt_path = self.dump_dir / ckpt_fname
            print(f"saving checkpoint to {ckpt_path}")
            with ckpt_path.open("wb") as f:
                torch.save(checkpoint, f)

    def load(self, ckpt_name: str = "ckpt.pt", device: str = "cpu") -> dict | None:
        path = self.dump_dir / ckpt_name
        if path.exists():
            with path.open("rb") as f:
                return torch.load(f, map_location=torch.device(device))
        else:
            return None
