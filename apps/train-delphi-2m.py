from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from delphi import distributed
from delphi.data import Dataset
from delphi.data.transform import TokenTransform
from delphi.data.ukb import UKBReader
from delphi.experiment import BaseTrainer, Logger, TrainBaseConfig, seed_everything
from delphi.log import Checkpointer
from delphi.model import Delphi2M, Delphi2MConfig


@dataclass
class TrainConfig(TrainBaseConfig):
    ckpt_dir: str = "debug"
    batch_size: int = 128
    seed: int = 42
    train_fold: str = "train"
    val_fold: str = "val"
    no_event_interval: None | float = 5.0 * 365.25
    no_event_mode: str = "legacy-random"
    exclude_lifestyle: bool = False
    augment_lifestyle: bool = False
    crop_mode: str = "left"
    fix_no_event_rate: bool = False
    break_clusters: bool = False
    additional_dx_token: bool = False
    model: Delphi2MConfig = field(
        default_factory=lambda: Delphi2MConfig(block_size=None, loss="homo_poisson")
    )


def train(cfg: TrainConfig):

    seed_everything(cfg.seed)

    reader = UKBReader()
    train_pids = UKBReader.participants(cfg.train_fold)
    val_pids = UKBReader.participants(cfg.val_fold)

    lifestyle_tokens = [reader.tokenizer[k] for k in reader.lifestyle_keys]
    sex_tokens = [reader.tokenizer[k] for k in reader.sex_keys]
    blacklist_tokens = lifestyle_tokens if cfg.exclude_lifestyle else None

    if cfg.additional_dx_token:
        dx_token = reader.vocab_size
        vocab_size = reader.vocab_size + 1
    else:
        dx_token = 1
        vocab_size = reader.vocab_size

    if cfg.break_clusters:
        whitelist_tokens = np.array([0, 1, dx_token] + sex_tokens + lifestyle_tokens)
    else:
        whitelist_tokens = None

    train_token_transform = TokenTransform(
        no_event_interval=cfg.no_event_interval,
        no_event_mode=cfg.no_event_mode,
        block_size=cfg.model.block_size,
        crop_mode=cfg.crop_mode,
        blacklist_tokens=blacklist_tokens,
        perturb_tokens=lifestyle_tokens if cfg.augment_lifestyle else None,
        break_clusters=cfg.break_clusters,
        dx_token=dx_token,
        whitelist_tokens=whitelist_tokens,
        seed=cfg.seed,
    )
    val_token_transform = train_token_transform.replace(perturb_tokens=None)

    train_ds = Dataset(
        reader=reader,
        pids=train_pids,
        token_transform=train_token_transform,
    )
    val_ds = Dataset(
        reader=reader,
        pids=val_pids,
        token_transform=val_token_transform,
    )

    cfg.model.vocab_size = vocab_size
    if cfg.additional_dx_token:
        cfg.model.self_terminate_except = list(
            set(cfg.model.self_terminate_except).union({dx_token})
        )
    if cfg.fix_no_event_rate:
        cfg.model.no_event_rate = 1 / cfg.no_event_interval

    if cfg.init_from == "scratch":
        print("initializing delphi-2m from scratch")
        model = Delphi2M(cfg.model)
    else:
        raise NotImplementedError

    backend = distributed.make_backend_from_args(cfg)
    cfg.wandb_project = cfg.ckpt_dir

    n_params = sum(p.numel() for p in model.parameters())
    logger = Logger(
        config=asdict(cfg),
        backend=backend,
        wandb_log=cfg.wandb_log,
        wandb_project=cfg.wandb_project,
        run_name=cfg.run_name,
        summary={"model_params": n_params},
    )

    metadata = {
        "config": asdict(cfg),
        "model_args": asdict(cfg.model),
        "token_transform_args": train_token_transform.config,
        "tokenizer": train_ds.tokenizer,
    }
    checkpointer = Checkpointer(
        dump_dir=Path(cfg.ckpt_dir) / logger.run_name,
        backend=backend,
        metadata=metadata,
    )

    trainer = BaseTrainer(
        cfg=cfg,
        backend=backend,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        logger=logger,
        checkpointer=checkpointer,
    )
    trainer.train()
    backend.finalize()


def main():

    default_cfg = OmegaConf.structured(TrainConfig())
    cli_args = OmegaConf.from_cli()

    if hasattr(cli_args, "config"):
        file_cfg = OmegaConf.load(cli_args.config)
        del cli_args.config
    else:
        file_cfg = default_cfg

    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)  # type: ignore


if __name__ == "__main__":
    main()
