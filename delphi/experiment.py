import copy
import json
import math
import os
import pprint
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from cloudpathlib import AnyPath
from omegaconf import OmegaConf
from typing_extensions import Self

from delphi import distributed
from delphi.env import DELPHI_CKPT_DIR
from delphi.log import Checkpointer, Logger, _format_for_display
from delphi.model.multimodal import DelphiM4, DelphiM4Config
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
class CliConfig:
    """Dataclass base providing OmegaConf-based CLI/config-file parsing.

    Subclasses get ``from_cli()`` for free. CLI syntax:

        python script.py config=foo.yaml key=value nested.key=value

    Precedence (low → high): dataclass defaults → YAML file → CLI args.
    The ``config`` key is reserved and cannot be used as a dataclass field.
    """

    @classmethod
    def from_cli(cls) -> Self:
        schema = OmegaConf.structured(cls)
        cli = OmegaConf.from_cli()
        if hasattr(cli, "config"):
            file_cfg = OmegaConf.load(cli.config)
            del cli.config
        else:
            file_cfg = OmegaConf.create({})
        merged = OmegaConf.merge(schema, file_cfg, cli)
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    def print(self):
        pprint.pprint(_format_for_display(asdict(self)))


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
    decay_iters: float | int = 0.1  # how many steps to decay for (wsd only)
    min_lr_frac: float = 0.1

    wandb_log: bool = True
    wandb_project: str = "delphi"
    tensorboard_log: bool = False
    tensorboard_dir: None | str = None  # local path; defaults to ./tb/<run_name>
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
        if self.device_type == "cuda" and cfg.dtype in ("float16", "bfloat16"):
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
            min_lr_frac=cfg.min_lr_frac,
            warmup_iters=cfg.warmup_iters,
            decay_iters=cfg.decay_iters,
            max_iters=cfg.max_iters,
            optimizer=self.optimizer,
        )
        self.scaler = torch.GradScaler(
            device=self.device_type, enabled=(cfg.dtype == "float16")
        )

        self.iter_num = 0
        ckpt_dict = checkpointer.load()
        if ckpt_dict is not None:
            if cfg.auto_resume:
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
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        batch_data = move_batch_to_device(args=batch_data, device=self.device)
        with self.ctx:
            output, loss, _ = self.model(*batch_data)

        return output, loss

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
                _, loss = self.mini_step(batch_data=batch_data)

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
                # push TB events to gs:// at the checkpoint cadence so curves are
                # visible mid-run (and survive a crash) on the ephemeral dsub VM
                self.logger.flush_to_gcs(self.checkpointer.dump_dir / "tb")
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
                    output, loss = self.mini_step(batch_data=batch_data)

                # backward pass, with gradient scaling if training in fp16.
                # divide by the accumulation steps so the accumulated gradient is
                # the mean (not the sum) of the micro-batches -> the effective
                # learning rate is independent of gradient_accumulation_steps.
                loss_agg = (
                    sum([loss[key] for key in loss.keys()])
                    / self.cfg.gradient_accumulation_steps
                )
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
                self.logger.log_grad_norm(self.model)
                self.logger.log_param_stats(self.model)
                self.logger.log_output(output)

            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            if hasattr(self.model, "update_ema"):
                self.model.update_ema()

            self.iter_num += 1
            # termination conditions
            if self.iter_num > self.cfg.max_iters:
                break

        # final push so the gs:// copy includes events logged after the last eval
        self.logger.flush_to_gcs(self.checkpointer.dump_dir / "tb")
        self.logger.finish()


@dataclass
class GenerateConfig(CliConfig):
    ckpt: str = "delphi-2m-og/ckpt.pt"
    # numeric age in years, or "recruitment" for per-participant recruitment age
    prompt_age: Any = None
    interval: float = 365.25
    batch_size: int = 512
    subsample: None | int = None
    n_repeats: int = 1
    stop_at_block_size: bool = True
    max_new_tokens: int = 128
    prompt_no_event: bool = False


def _backfill_biomarker2idx(model_args: dict) -> None:
    """Reconstruct biomarker2idx for legacy checkpoints that predate the field.

    Old checkpoints indexed modalities by the Modality enum values, so the saved
    mod_embedding weights and bio_m channel are keyed by those values. Rebuild the
    same mapping from the enum so the loaded model (and BiomarkerTransform.from_ckpt)
    agree with the trained weights. Mutates model_args in place.
    """
    biomarkers = model_args.get("biomarkers") or {}
    if not biomarkers or model_args.get("biomarker2idx"):
        return
    from delphi.multimodal import Modality  # local: legacy-only dependency

    try:
        model_args["biomarker2idx"] = {
            name: Modality[name.upper()].value for name in biomarkers
        }
    except KeyError as e:
        raise ValueError(
            f"legacy checkpoint references biomarker {e} not in the Modality enum; "
            "cannot reconstruct biomarker2idx"
        )
    print("[load_ckpt] back-filled biomarker2idx from Modality enum (legacy ckpt)")


def _backfill_reader_args(ckpt_dict: dict) -> None:
    """Synthesize reader_args for legacy checkpoints that stored a combined data_args.

    Older checkpoints saved a single data_args blob instead of today's split
    reader_args / token_transform_args / biomarker_transform_args. reader_args only
    carries the biomarker and expansion-pack lists, both already present in data_args.
    Mutates ckpt_dict in place. (token/biomarker transform args are not reconstructed:
    name mismatches and the missing biomarker_stats make that lossy; out of scope here.)
    """
    if "reader_args" in ckpt_dict or "data_args" not in ckpt_dict:
        return
    data_args = ckpt_dict["data_args"]
    ckpt_dict["reader_args"] = {
        "biomarkers": data_args.get("biomarkers"),
        "expansion_packs": data_args.get("expansion_packs"),
    }
    print("[load_ckpt] synthesized reader_args from legacy data_args")


# Legacy DelphiM4 used a nested transformer.embed sub-module bundling all input
# embeddings; the current model flattens these out. Map old keys -> new keys by
# longest-matching prefix substitution. Order doesn't matter (all prefixes are
# disjoint after the common 'transformer.embed.' root).
_LEGACY_STATE_DICT_REMAP = [
    ("transformer.embed.token_embedding.", "transformer.wte."),
    ("transformer.embed.age_encoding.", "transformer.wae."),
    ("transformer.embed.biomarker_embed.", "bio_embed.embed."),
    ("transformer.embed.mod_embedding.", "mod_embedding."),
]


def _remap_legacy_state_dict(state_dict: dict) -> dict:
    """Translate state_dict keys from the legacy nested transformer.embed layout
    to the current flattened model. Returns a new dict; no-op if no legacy keys
    are present.
    """
    out, remapped = {}, 0
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in _LEGACY_STATE_DICT_REMAP:
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix) :]
                remapped += 1
                break
        out[new_k] = v
    if remapped:
        print(
            f"[load_ckpt] remapped {remapped} legacy state_dict keys "
            "(transformer.embed.* -> current layout)"
        )
    return out


# Delphi2M (the retired unimodal model) is the zero-biomarker special case of
# DelphiM4: the two build an identical transformer/lm_head state_dict (verified
# key-for-key) and DelphiM4's forward reduces to Delphi2M's when no biomarkers are
# passed. So a delphi-2m checkpoint loads by translating its config into a
# zero-biomarker DelphiM4Config; the weights then transfer 1:1 via load_state_dict.
_DELPHI2M_LOSS_REMAP = {
    # delphi-2m loss name -> delphi-m4 loss name. "default" and "homo_poisson" both
    # produce logits via lm_head and are read through HomoPoissonTPP at inference,
    # so they map to the same M4 head with identical forward numerics.
    "default": "homo_poisson",
    "homo_poisson": "homo_poisson",
}


def _upgrade_delphi2m_model_args(model_args: dict) -> dict:
    """Translate delphi-2m model_args into zero-biomarker delphi-m4 model_args.

    Only fields shared with DelphiM4Config survive load_ckpt's later field filter,
    so this just remaps the loss name, clears biomarkers, and rejects the two
    settings DelphiM4 cannot reproduce (cluster-poisson has no M4 head;
    mask_no_event_attention has no M4 equivalent).
    """
    out = dict(model_args)
    if out.get("mask_no_event_attention"):
        raise ValueError(
            "cannot load delphi-2m checkpoint with mask_no_event_attention=True: "
            "DelphiM4 has no equivalent attention-masking flag"
        )
    loss = out.get("loss", "default")
    if loss not in _DELPHI2M_LOSS_REMAP:
        raise ValueError(
            f"cannot load delphi-2m checkpoint with loss={loss!r} into DelphiM4 "
            f"(supported: {sorted(_DELPHI2M_LOSS_REMAP)}); "
            "e.g. homo_cluster_poisson is not wired into DelphiM4"
        )
    out["loss"] = _DELPHI2M_LOSS_REMAP[loss]
    out["biomarkers"] = {}
    out["biomarker2idx"] = {}
    return out


def load_ckpt(ckpt_path):

    ckpt_path = AnyPath(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with ckpt_path.open("rb") as f:
        ckpt_dict = torch.load(f, map_location=device)
    model_type = ckpt_dict["model_type"]
    if model_type == "delphi-2m":
        # Delphi2M was retired; load its checkpoints as a zero-biomarker DelphiM4
        # (identical transformer/lm_head weights and forward numerics).
        print(
            "[load_ckpt] upgrading delphi-2m checkpoint to DelphiM4 (zero biomarkers)"
        )
        ckpt_dict["model_args"] = _upgrade_delphi2m_model_args(ckpt_dict["model_args"])
        ckpt_dict.setdefault(
            "reader_args", {"biomarkers": None, "expansion_packs": None}
        )
        model_type = "delphi-m4"

    if model_type == "delphi-m4":
        model_cfg_cls = DelphiM4Config
        model_cls = DelphiM4
    else:
        raise ValueError(
            f"unsupported model_type {model_type!r}; expected 'delphi-m4' "
            "(or 'delphi-2m', which is upgraded to DelphiM4 on load)"
        )

    _backfill_biomarker2idx(ckpt_dict["model_args"])
    _backfill_reader_args(ckpt_dict)

    pprint.pp(ckpt_dict["model_args"])
    valid_fields = {f.name for f in fields(model_cfg_cls)}
    model_args = {k: v for k, v in ckpt_dict["model_args"].items() if k in valid_fields}
    model_cfg = model_cfg_cls(**model_args)
    model = model_cls(model_cfg)  # type: ignore
    ckpt_dict["model"] = _remap_legacy_state_dict(ckpt_dict["model"])
    missing, unexpected = model.load_state_dict(ckpt_dict["model"], strict=False)
    if missing:
        raise RuntimeError(
            f"checkpoint missing {len(missing)} required parameter(s); these would "
            f"be silently random-initialized: {missing}"
        )
    if unexpected:
        print(f"[load_ckpt] {len(unexpected)} unexpected key(s) ignored: {unexpected}")
    model.to(device)
    model = model.eval()

    return model, ckpt_dict


def flexi_list(panel):
    if isinstance(panel, str):
        if panel.endswith(".yaml"):
            with open(panel, "r") as f:
                return yaml.safe_load(f)
        else:
            return [panel]
    elif isinstance(panel, list):
        return panel
    else:
        raise ValueError


def match_unique(query, choices, *, key=None, label="value"):
    """Resolve a CLI argument to the one element of ``choices`` whose text
    contains ``query`` as a case-insensitive substring.

    ``key`` maps a choice to the string searched (default: the choice itself), so
    callers can match against a derived/concatenated field. Raises ``SystemExit``
    (a clean CLI message, no traceback) when the query is ambiguous (>1) or
    unmatched (0), listing the matches so it can't silently mis-pick. An exact
    full name still matches only itself.
    """
    choices = list(choices)
    to_text = key or (lambda c: c)
    q = query.lower()
    matches = [c for c in choices if q in str(to_text(c)).lower()]
    if len(matches) != 1:
        raise SystemExit(
            f"{label}={query!r} matched {len(matches)} of {len(choices)} candidates "
            f"{sorted(map(str, matches))}; need exactly one — be more specific"
        )
    return matches[0]


def load_json(json_path):
    with AnyPath(json_path).open() as f:
        data = json.load(f)
        if "config" in data.keys():
            config = copy.deepcopy(data["config"])
            del data["config"]
        else:
            config = None
    return data, config
