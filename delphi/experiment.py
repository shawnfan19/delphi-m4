import math
import os
import pprint
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Self

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from delphi import distributed
from delphi.env import DELPHI_CKPT_DIR
from delphi.log import Checkpointer, Logger
from delphi.model.multimodal import DelphiM4, DelphiM4Config
from delphi.model.transformer import Delphi2M, Delphi2MConfig
from delphi.optim import (
    configure_optimizer,
    configure_scheduler,
    parse_weight_decay_groups,
)


# Update this function whenever you have a library that needs to be seeded.
def seed_everything(seed):
    """Seed all random generators."""
    random.seed(seed)

    # For numpy:
    # This is for legacy numpy:
    np.random.seed(seed)
    # New code should make a Generator out of the config.seed directly:
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html

    # For PyTorch:
    torch.manual_seed(seed)

    # if config.cuda_deterministic:
    #     # Higher (e.g., on CUDA too) reproducibility with deterministic algorithms:
    #     # https://pytorch.org/docs/stable/notes/randomness.html
    #
    #     # Not supported for all operations though:
    #     # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    #     if config.cuda_strong_deterministic:
    #         torch.use_deterministic_algorithms(True)
    #
    #     #  A lighter version of the above otherwise as not all algorithms have a deterministic implementation
    #     torch.backends.cudnn.deterministic = True
    #
    #     # torch.backends.cudnn.benchmark = False
    #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def move_batch_to_device(args: Iterable, device: str | torch.device):

    outputs = list()
    for arg in args:
        if isinstance(arg, torch.Tensor):
            outputs.append(arg.to(device))
        elif isinstance(arg, dict):
            outputs.append({k: v.to(device) for k, v in arg.items()})
        else:
            raise NotImplementedError
    return tuple(outputs)


def fixed_batch_iter(seed: int, batch_size: int, total_size: int):
    rng = np.random.default_rng(seed)
    batch_size = min(batch_size, total_size)
    batch_idx = rng.integers(total_size, size=(batch_size,))
    while True:
        yield batch_idx


def train_iter(
    seed: int,
    total_size: int,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    step: int = 0,
) -> Iterator[np.ndarray]:

    while True:
        seed_with_offset = seed + step * world_size + rank
        rng = np.random.default_rng(seed_with_offset)
        batch_idx = rng.integers(total_size, size=(batch_size,))
        step += 1

        yield batch_idx


def eval_iter(total_size: int, batch_size: int) -> Iterator[np.ndarray]:

    batch_start_pos = np.arange(0, total_size, batch_size)
    batch_end_pos = batch_start_pos + batch_size
    batch_end_pos[-1] = total_size

    for start, end in zip(batch_start_pos, batch_end_pos):
        yield np.arange(start, end)


@dataclass
class TrainBaseConfig:
    ckpt_dir: str = "debug"
    eval_interval: int = 2000
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    init_from: str = "scratch"
    auto_resume: bool = True

    debug_batch: bool = False

    seed: int = 42
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes
    batch_size: int = 128
    # if gradient_accumulation_steps > 1, this is the micro-batch size

    # system
    device: str = "cuda"
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float32"
    # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

    distributed_backend: Optional[str] = None

    # adamw optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 100000  # total number of training iterations
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    schedule: str = "cosine"  # consine, constant
    warmup_iters: float | int = 1000  # how many steps to warm up for
    min_lr: float = 0.1

    wandb_log: bool = True
    wandb_project: str = "delphi"
    run_name: None | str = None
    ckpt_interval: None | int = None
    log_interval: int = 250


class BaseTrainer:

    def __init__(
        self,
        cfg: TrainBaseConfig,
        backend: distributed.backend.DistributedBackend,
        model: torch.nn.Module,
        train_ds: Any,
        val_ds: Any,
        logger: Logger,
        checkpointer: Checkpointer,
        optimizer: None | torch.optim.Optimizer = None,
    ):
        self.backend = backend
        cfg = self.backend.get_adjusted_args_for_process(cfg)

        self.cfg = cfg
        self.device = cfg.device
        self.device_type = (
            "cuda" if "cuda" in cfg.device else "cpu"
        )  # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        self.ptdtype = {
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg.dtype]
        if self.device_type == "cuda" and self.ptdtype == "float16":
            self.ctx = torch.autocast(device_type=self.device_type, dtype=self.ptdtype)
        else:
            self.ctx = nullcontext()

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model = model
        self.model.to(self.device)
        param_groups = parse_weight_decay_groups(model=self.model)

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = configure_optimizer(
                optim_groups=param_groups,
                learning_rate=cfg.learning_rate,
                beta1=cfg.beta1,
                beta2=cfg.beta2,
            )

        self.scheduler = configure_scheduler(
            schedule=cfg.schedule,
            learning_rate=cfg.learning_rate,
            min_lr=cfg.min_lr,
            warmup_iters=cfg.warmup_iters,
            max_iters=cfg.max_iters,
            optimizer=self.optimizer,
        )
        self.scaler = torch.GradScaler(
            device=self.device_type, enabled=(cfg.dtype == "float16")
        )

        self.iter_num = 0
        if cfg.auto_resume:
            ckpt_dict = checkpointer.load()
            if ckpt_dict is not None:
                self.model.load_state_dict(ckpt_dict["model"])
                self.optimizer.load_state_dict(ckpt_dict["optimizer"])
                self.scheduler.load_state_dict(ckpt_dict["scheduler"])
                self.iter_num = ckpt_dict["iter_num"]
                if self.backend.is_master_process():
                    print(
                        f"found and loaded existing checkpoint; starting from iter {self.iter_num}"
                    )
        self.model = self.backend.transform_model(self.model)

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(f"\tinitialized data loader for worker {self.rank}/{self.world_size}")
        else:
            self.world_size = 1
            self.rank = 0

        if cfg.debug_batch:
            self.train_iter = fixed_batch_iter(
                seed=cfg.seed, batch_size=cfg.batch_size, total_size=len(train_ds)
            )
        else:
            self.train_iter = train_iter(
                seed=cfg.seed,
                total_size=len(train_ds),
                batch_size=cfg.batch_size,
                world_size=self.world_size,
                rank=self.rank,
                step=self.iter_num,
            )

        self.logger = logger
        self.checkpointer = checkpointer
        self.best_val_loss = float("inf")

    def mini_step(
        self, batch_data: Iterable, *args, **kwargs
    ) -> dict[str, torch.Tensor]:

        batch_data = move_batch_to_device(args=batch_data, device=self.device)
        with self.ctx:
            _, loss, _ = self.model(*batch_data)

        return loss

    @torch.no_grad()
    def estimate_loss(self, *args, **kwargs) -> dict:
        training_states = {
            name: mod.training for name, mod in self.model.named_modules()
        }
        self.model.eval()  # type: ignore
        eval_loss = {}
        for split in ["train", "val"]:
            eval_ds = self.train_ds if split == "train" else self.val_ds

            if self.cfg.debug_batch:
                estimate_iter = fixed_batch_iter(
                    seed=self.cfg.seed,
                    total_size=len(eval_ds),
                    batch_size=self.cfg.batch_size,
                )
                eval_iters = 1
            else:
                estimate_iter = train_iter(
                    seed=self.cfg.seed,
                    total_size=len(eval_ds),
                    batch_size=self.cfg.batch_size,
                    world_size=self.world_size,
                    rank=self.rank,
                )
                eval_iters = min(
                    self.cfg.eval_iters, math.ceil(len(eval_ds) / self.cfg.batch_size)
                )
            split_loss = defaultdict(float)
            for _ in range(eval_iters):

                batch_idx = next(estimate_iter)
                batch_data = eval_ds.get_batch(batch_idx)
                loss = self.mini_step(batch_data=batch_data)

                for key in loss.keys():
                    split_loss[key] += loss[key].item()
            split_loss = dict(split_loss)
            eval_loss[f"{split}/loss"] = 0
            for key in split_loss.keys():
                eval_loss[f"{split}/{key}"] = split_loss[key] / eval_iters
                eval_loss[f"{split}/loss"] += eval_loss[f"{split}/{key}"]

        for name, mod in self.model.named_modules():
            mod.training = training_states[name]

        return eval_loss

    def _save_ckpt(self, step: int, ckpt_fname: str = "ckpt.pt"):
        self.checkpointer.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            best_val_loss=self.best_val_loss,
            ckpt_fname=ckpt_fname,
        )

    def train(self):

        if self.cfg.compile:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)

        while True:

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.cfg.eval_interval == 0:
                eval_loss = self.estimate_loss()
                metrics = {"step": self.iter_num} | eval_loss
                self.logger.log(metrics)
                self.logger.print(
                    f"iter {self.iter_num}: "
                    f"train loss {eval_loss['train/loss']:.4f}, "
                    f"val loss {eval_loss['val/loss']:.4f}"
                )
                self._save_ckpt(self.iter_num, ckpt_fname="ckpt.pt")
                if eval_loss["val/loss"] < self.best_val_loss:
                    self._save_ckpt(self.iter_num, ckpt_fname="ckpt_best.pt")
                self.best_val_loss = min(eval_loss["val/loss"], self.best_val_loss)

            if self.cfg.ckpt_interval is not None:
                if self.iter_num % self.cfg.ckpt_interval == 0:
                    self._save_ckpt(
                        self.iter_num, ckpt_fname=f"ckpt_{self.iter_num}.pt"
                    )

            if self.iter_num == 0 and self.cfg.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for i in range(self.cfg.gradient_accumulation_steps):
                with self.backend.get_context_for_microstep_forward(
                    model=self.model,
                    microstep_idx=i,
                    gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
                ):
                    batch_idx = next(self.train_iter)
                    batch_data = self.train_ds.get_batch(batch_idx)
                    loss = self.mini_step(batch_data=batch_data)

                # backward pass, with gradient scaling if training in fp16
                loss_agg = sum([loss[key] for key in loss.keys()])
                self.scaler.scale(loss_agg).backward()  # type: ignore

            # clip the gradient
            if self.cfg.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )

            # log training metrics
            if self.iter_num % self.cfg.log_interval == 0:
                metrics = {
                    "step": self.iter_num,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                lossf = 0.0
                for loss_key, loss_pt in loss.items():
                    metrics[loss_key] = loss_pt.item()
                    lossf += loss_pt.item()
                metrics["loss"] = lossf

                self.logger.print(f"iter {self.iter_num}: loss {lossf:.4f}")
                self.logger.log(metrics)
                self.logger.log_grad(self.model)

            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            self.iter_num += 1
            # termination conditions
            if self.iter_num > self.cfg.max_iters:
                break


@dataclass
class GenerateConfig:
    ckpt: str = "delphi-2m-og/ckpt.pt"
    prompt_age: None | int | float = None
    prompt_lifestyle: bool = True
    interval: float = 365.25
    batch_size: int = 512
    subsample: None | int = None
    n_repeats: int = 1
    stop_at_block_size: bool = True
    max_new_tokens: int = 128
    prompt_no_event: bool = False

    @classmethod
    def from_cli(cls) -> Self:
        """Parse from command line arguments"""
        # Create structured config from dataclass defaults
        schema = OmegaConf.structured(cls)
        # Parse CLI args (format: key=value)
        cli = OmegaConf.from_cli()
        # Merge: CLI overrides defaults
        merged = OmegaConf.merge(schema, cli)
        # Convert back to dataclass instance
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    @classmethod
    def auto(cls, **overrides) -> Self:
        """
        Automatically choose:
        - Interactive env → use defaults + overrides
        - CLI → parse arguments
        """
        if "ipykernel" in sys.modules or "IPython" in sys.modules:
            print("Running in interactive environment")
            schema = OmegaConf.structured(cls)
            override_conf = OmegaConf.create(overrides)
            merged = OmegaConf.merge(schema, override_conf)
            return OmegaConf.to_object(merged)  # type: ignore[return-value]
        else:
            return cls.from_cli()


def load_ckpt(ckpt_path):

    ckpt_path = Path(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    model_type = ckpt_dict["model_type"]
    if model_type == "delphi-2m":
        model_cfg_cls = Delphi2MConfig
        model_cls = Delphi2M
    elif model_type == "delphi-m4":
        model_cfg_cls = DelphiM4Config
        model_cls = DelphiM4
    else:
        raise ValueError

    pprint.pp(ckpt_dict["model_args"])
    valid_fields = {f.name for f in fields(model_cfg_cls)}
    model_args = {k: v for k, v in ckpt_dict["model_args"].items() if k in valid_fields}
    model_cfg = model_cfg_cls(**model_args)
    model = model_cls(model_cfg)  # type: ignore
    model.load_state_dict(ckpt_dict["model"], strict=False)
    missing, unexpected = model.load_state_dict(ckpt_dict["model"], strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    model.to(device)
    model = model.eval()

    return model, ckpt_dict
