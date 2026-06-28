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


# The logging backends Logger can select, via its `log_backend` arg (fed by the
# DELPHI_LOG_BACKEND env / `log_backend=` config). "none" disables logging.
LOG_BACKENDS = ("wandb", "tensorboard", "trackio", "none")


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
        # Mirror the offline run dir so the run is recoverable / `wandb sync`-able
        # even if the offline, ephemeral job is killed. All runs go under ONE
        # root-level dir, $DELPHI_CKPT_DIR/wandb/<offline-run>: wandb groups by
        # project/name in its UI, so the on-disk location is just for bulk sync.
        # base_dir (the run dir) is used only to detect gs:// mode. No-op when
        # online (data already streams to the cloud) or for a local base_dir.
        if not isinstance(base_dir, CloudPath):
            return
        if os.environ.get("WANDB_MODE", "online") not in ("offline", "dryrun"):
            return
        run_dir = os.path.dirname(self.wandb.run.dir)
        # force overwrite: we own this path and re-upload it every flush; without
        # it cloudpathlib refuses static files (e.g. requirements.txt) whose cloud
        # copy from a prior flush is newer than the unchanged local copy.
        (AnyPath(DELPHI_CKPT_DIR) / "wandb" / os.path.basename(run_dir)).upload_from(
            run_dir, force_overwrite_to_cloud=True
        )

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
        # base_dir is the run dir <root>/<group>/<run>; put TB at the GROUP level
        # as <group>/tb/<run> (.parent = group, .name = run) so a single
        # `tensorboard --logdir <group>/tb` shows every run in the group as its
        # own series. No-op for a local base_dir -- the events persist on disk.
        self.writer.flush()
        if isinstance(base_dir, CloudPath):
            # force overwrite: re-uploaded each flush (see WandbBackend.flush_to_gcs)
            (base_dir.parent / "tb" / base_dir.name).upload_from(
                self.log_dir, force_overwrite_to_cloud=True
            )

    def finish(self):
        self.writer.close()


class TrackioBackend:
    """Logs scalar dicts to a local trackio SQLite db (works offline) and, when
    nvidia-ml-py / psutil are present, auto-collects GPU + CPU/RAM metrics on a
    background thread -- the thing TensorBoard can't do for you.

    Stays purely local: no space_id / server_url, so trackio.init() writes to
    TRACKIO_DIR/<project>.db with no network call and starts no server (the
    Gradio dashboard runs only via `trackio show`, e.g. on a laptop after
    pulling the db). flush_to_gcs mirrors that single db file under gs:// so a
    killed offline job stays recoverable.
    """

    def __init__(self, project, run_name, config, summary):
        import trackio
        from trackio.sqlite_storage import SQLiteStorage

        self.trackio = trackio
        # Force local-only logging. trackio.init() reads TRACKIO_SPACE_ID /
        # TRACKIO_SERVER_URL / TRACKIO_BUCKET_ID from the env and flips to remote
        # mode (HF/RemoteClient network calls that hang behind the VPC-SC
        # perimeter) if any are set -- and `None or os.environ.get(...)` picks
        # them up even though we pass no space_id. Drop them so the offline VM
        # never reaches a remote code path regardless of inherited environment.
        for var in ("TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_BUCKET_ID"):
            os.environ.pop(var, None)
        # embed=False: never embed/launch a dashboard (no-op off a notebook, but
        # explicit on the headless VM). summary is folded into config -- trackio
        # has no wandb-style run summary, and these (e.g. model_params) are
        # constants that belong in the config panel anyway.
        trackio.init(
            project=project,
            name=run_name,
            config={**config, **(summary or {})},
            embed=False,
        )
        # same `project` string trackio.init used, so trackio's own filename
        # sanitization makes this path agree with where it actually wrote.
        self.db_path = SQLiteStorage.get_project_db_path(project)

    def log(self, metrics, step, commit):
        # trackio.log has no `commit` (a wandb batching concept); each call writes
        # a row at `step`. Drop a redundant in-dict "step" -- it's already carried
        # by step=, and trackio reserves the key (it would rename it to "__step"
        # and warn on every call). Distinct grad/param/output keys keep their own
        # columns at the same step -- no clobber.
        metrics = {k: v for k, v in metrics.items() if k != "step"}
        self.trackio.log(metrics, step=step)

    def flush_to_gcs(self, base_dir):
        # One SQLite db per project at DELPHI_CKPT_DIR/trackio/<project>.db
        # (root-level like wandb; trackio groups runs by project/name internally).
        # Pull it into a laptop's TRACKIO_DIR and `trackio show --project <name>`.
        # No-op for a local base_dir -- the db persists on disk.
        if not isinstance(base_dir, CloudPath):
            return
        if self.db_path.exists():
            # force overwrite: re-uploaded each flush (see WandbBackend.flush_to_gcs)
            (AnyPath(DELPHI_CKPT_DIR) / "trackio" / self.db_path.name).upload_from(
                str(self.db_path), force_overwrite_to_cloud=True
            )

    def finish(self):
        self.trackio.finish()


class Logger:
    """Receives metric dicts and sends them to ONE backend (wandb, tensorboard,
    or trackio) + stdout.

    Exactly one backend, chosen by `log_backend` -- the shared master-process
    gating and metric construction live here so the backends stay tiny.
    TensorBoard needs
    an explicit global_step, so we track the latest "step" seen (logged before
    the commit=False grad/param/output stats in BaseTrainer.train, so it is
    current when those fire); wandb gets the step implicitly via step_metric.
    """

    def __init__(
        self,
        config: dict,
        backend: distributed.backend.DistributedBackend,
        log_backend: str = "wandb",
        logger_project: str = "delphi",
        tensorboard_dir: None | str = None,
        run_name: None | str = None,
        summary: None | dict = None,
    ):
        if log_backend not in LOG_BACKENDS:
            raise ValueError(
                f"log_backend must be one of {list(LOG_BACKENDS)}, got {log_backend!r}"
            )
        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.run_name = run_name
        self.backend = backend
        self._step = 0
        # the active backend instance (or None for "none" / non-master)
        self.log_backend = None

        if backend.is_master_process():
            print("=== config ===")
            pprint(_format_for_display(config), indent=2, width=60)

            if log_backend == "wandb":
                self.log_backend = WandbBackend(
                    logger_project, run_name, config, summary
                )
            elif log_backend == "tensorboard":
                if tensorboard_dir is None:
                    tensorboard_dir = os.path.join("tb", run_name)
                self.log_backend = TensorBoardBackend(tensorboard_dir)
            elif log_backend == "trackio":
                # logger_project is the project name (the train apps set it to
                # ckpt_dir); trackio groups runs under it in the local db.
                self.log_backend = TrackioBackend(
                    logger_project, run_name, config, summary
                )
            # log_backend == "none": no backend, stdout config dump only

    def _emit(self, metrics: dict, commit: bool = True):
        if self.log_backend is None:  # non-master, or no backend enabled
            return
        if "step" in metrics:
            self._step = metrics["step"]
        self.log_backend.log(metrics, step=self._step, commit=commit)

    def log(self, metrics: dict):
        self._emit(metrics, commit=True)

    def log_grad_norm(self, model: torch.nn.Module):
        if self.log_backend is None:
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                self._emit(
                    {f"grad_norm/{name}": param.grad.norm().item()}, commit=False
                )

    def log_param_stats(self, model: torch.nn.Module):
        if self.log_backend is None:
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
        if self.log_backend is None:
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
        if self.log_backend is not None:
            self.log_backend.flush_to_gcs(base_dir)

    def finish(self):
        if self.log_backend is not None:
            self.log_backend.finish()

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
