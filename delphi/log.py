import os
from datetime import datetime
from pathlib import Path
from pprint import pprint

import torch
import wandb

from delphi import distributed
from delphi.env import DELPHI_CKPT_DIR


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
            pprint(config, indent=2, width=60)

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

                if summary is not None:
                    for k, v in summary.items():
                        wandb.summary[k] = v

    def log(self, metrics: dict):
        if self.backend.is_master_process():
            if self.wandb:
                wandb.log(metrics)

    def log_grad(self, model: torch.nn.Module):
        if self.backend.is_master_process():
            if self.wandb:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log(
                            {f"grad_norm/{name}": param.grad.norm().item()},
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
        self.dump_dir = Path(DELPHI_CKPT_DIR) / dump_dir
        self.backend = backend
        self.metadata = metadata or {}

        if backend.is_master_process():
            os.makedirs(self.dump_dir, exist_ok=True)

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
            ckpt_path = os.path.join(self.dump_dir, ckpt_fname)
            print(f"saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)

    def load(self, ckpt_name: str = "ckpt.pt", device: str = "cpu") -> dict | None:
        path = self.dump_dir / ckpt_name
        if os.path.exists(path):
            return torch.load(
                self.dump_dir / ckpt_name, map_location=torch.device(device)
            )
        else:
            return None
