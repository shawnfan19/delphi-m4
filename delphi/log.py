import os
from datetime import datetime
from pprint import pprint

import torch
from cloudpathlib import AnyPath, CloudPath

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


class WandbBackend:
    """Sends scalar dicts to Weights & Biases (needs internet)."""

    def __init__(self, project, run_name, config, summary):
        import re

        import wandb

        self.wandb = wandb
        project = re.sub(r"[^a-zA-Z0-9_\-.]", "_", project)
        wandb.init(project=project, name=run_name, config=config)
        wandb.define_metric("step")
        for glob in ("lr", "val/*", "train/*", "grad_norm/*", "output/*", "param/*"):
            wandb.define_metric(glob, step_metric="step")
        if summary is not None:
            for k, v in summary.items():
                wandb.summary[k] = v

    def log(self, metrics, step, commit):
        self.wandb.log(metrics, commit=commit)

    def flush_to_gcs(self, base_dir):
        # Mirror the offline run dir to <base_dir>/wandb on gs:// mid-run so the
        # run is recoverable / `wandb sync`-able even if the offline, ephemeral
        # job is killed before exit. No-op when online (data already streams to
        # the wandb cloud) or for a local base_dir (the dir already persists).
        if not isinstance(base_dir, CloudPath):
            return
        if os.environ.get("WANDB_MODE", "online") not in ("offline", "dryrun"):
            return
        run_dir = os.path.dirname(self.wandb.run.dir)
        (base_dir / "wandb" / os.path.basename(run_dir)).upload_from(run_dir)

    def finish(self):
        self.wandb.finish()


class TensorBoardBackend:
    """Writes scalar dicts to local TensorBoard event files (works offline).

    SummaryWriter uses plain file ops, so log_dir must be a LOCAL path -- not a
    gs:// cloudpath like the Checkpointer accepts. On dsub point it at the
    --output mount so the events get synced to GCS for viewing.
    """

    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = str(log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, metrics, step, commit):
        for k, v in metrics.items():
            if k != "step" and isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)

    def flush_to_gcs(self, base_dir):
        # Push event files to <base_dir>/tb on gs:// mid-run so curves are
        # visible before the (offline, ephemeral) job exits. No-op for a local
        # base_dir -- there the event dir already persists on disk.
        self.writer.flush()
        if isinstance(base_dir, CloudPath):
            (base_dir / "tb").upload_from(self.log_dir)

    def finish(self):
        self.writer.close()


class Logger:
    """Receives metric dicts and sends them to ONE backend (wandb OR
    tensorboard) + stdout.

    Exactly one backend at a time -- the shared master-process gating and
    metric construction live here so the backends stay tiny. TensorBoard needs
    an explicit global_step, so we track the latest "step" seen (logged before
    the commit=False grad/param/output stats in BaseTrainer.train, so it is
    current when those fire); wandb gets the step implicitly via step_metric.
    """

    def __init__(
        self,
        config: dict,
        backend: distributed.backend.DistributedBackend,
        wandb_log: bool = True,
        wandb_project: str = "delphi",
        tensorboard_log: bool = False,
        tensorboard_dir: None | str = None,
        run_name: None | str = None,
        summary: None | dict = None,
    ):
        if wandb_log and tensorboard_log:
            raise ValueError(
                "enable only one logging backend (wandb_log or tensorboard_log)"
            )
        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.run_name = run_name
        self.backend = backend
        self._step = 0
        self._impl = None  # set on the master process only

        if backend.is_master_process():
            print("=== config ===")
            pprint(_format_for_display(config), indent=2, width=60)

            if wandb_log:
                self._impl = WandbBackend(wandb_project, run_name, config, summary)
            elif tensorboard_log:
                if tensorboard_dir is None:
                    tensorboard_dir = os.path.join("tb", run_name)
                self._impl = TensorBoardBackend(tensorboard_dir)

    def _emit(self, metrics: dict, commit: bool = True):
        if self._impl is None:  # non-master, or no backend enabled
            return
        if "step" in metrics:
            self._step = metrics["step"]
        self._impl.log(metrics, step=self._step, commit=commit)

    def log(self, metrics: dict):
        self._emit(metrics, commit=True)

    def log_grad_norm(self, model: torch.nn.Module):
        if self._impl is None:
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                self._emit(
                    {f"grad_norm/{name}": param.grad.norm().item()}, commit=False
                )

    def log_param_stats(self, model: torch.nn.Module):
        if self._impl is None:
            return
        for name, param in model.named_parameters():
            tensor = param.detach().float()
            self._emit(
                {
                    f"param/{name}/mean": tensor.mean().item(),
                    f"param/{name}/median": tensor.median().item(),
                    f"param/{name}/max": tensor.max().item(),
                    f"param/{name}/min": tensor.min().item(),
                },
                commit=False,
            )

    def log_output(self, output: dict[str, torch.Tensor]):
        if self._impl is None:
            return
        for name, tensor in output.items():
            if isinstance(tensor, dict):
                continue
            if not torch.is_floating_point(tensor):
                continue
            tensor = tensor.detach().float()
            self._emit(
                {
                    f"output/{name}/mean": tensor.mean().item(),
                    f"output/{name}/median": tensor.median().item(),
                    f"output/{name}/max": tensor.max().item(),
                    f"output/{name}/min": tensor.min().item(),
                },
                commit=False,
            )

    def flush_to_gcs(self, base_dir):
        """Push the active backend's local artifacts under base_dir on gs://
        mid-run (called at the checkpoint cadence). The backend picks its own
        subdir (tb/ or wandb/). No-op on non-master / when no backend is active.
        """
        if self._impl is not None:
            self._impl.flush_to_gcs(base_dir)

    def finish(self):
        if self._impl is not None:
            self._impl.finish()

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
