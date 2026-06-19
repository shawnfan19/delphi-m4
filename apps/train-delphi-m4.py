from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from delphi import distributed
from delphi.data import MultimodalDataset
from delphi.data.transform import BiomarkerTransform, TokenTransform
from delphi.data.ukb import Biomarker, MultimodalUKBReader
from delphi.experiment import BaseTrainer, Logger, TrainBaseConfig, seed_everything
from delphi.log import Checkpointer
from delphi.model.multimodal import DelphiM4, DelphiM4Config
from delphi.multimodal import compose_panel


@dataclass
class TrainConfig(TrainBaseConfig):
    ckpt_dir: str = "debug"
    batch_size: int = 128
    seed: int = 42
    deterministic: bool = False
    train_fold: str = "train"
    val_fold: str = "val"
    model: DelphiM4Config = field(
        default_factory=lambda: DelphiM4Config(block_size=None)
    )
    panel: Any = None
    biomarkers: None | list[str] = None
    first_time_only: bool = True
    must_have: bool = False
    z_score_biomarkers: bool = True
    expansion_packs: None | list[str] = None
    must_have_expansion_packs: bool = False
    ignore_expansion_packs: bool = True
    biomarker_dropout: None | float = None
    exclude_smoking_and_alcohol: bool = False
    tiebreak: bool = False

    def __post_init__(self):
        if self.panel:
            self.biomarkers, self.expansion_packs, _ = compose_panel(
                self.panel, self.biomarkers, self.expansion_packs
            )


def train(cfg: TrainConfig):

    seed_everything(cfg.seed)

    train_pids = MultimodalUKBReader.participants(cfg.train_fold)
    val_pids = MultimodalUKBReader.participants(cfg.val_fold)
    if cfg.must_have:
        print(f"keeping participants with any of: {cfg.biomarkers}")
        total_train, total_val = train_pids.size, val_pids.size
        train_pids = MultimodalUKBReader.filter_participants_with_biomarkers(
            train_pids, biomarkers=cfg.biomarkers, any=True
        )
        val_pids = MultimodalUKBReader.filter_participants_with_biomarkers(
            val_pids, biomarkers=cfg.biomarkers, any=True
        )
        print(f"{train_pids.size} / {total_train} train pids")
        print(f"{val_pids.size} / {total_val} val pids")

    if cfg.must_have_expansion_packs:
        print(f"keeping participants with any of: {cfg.expansion_packs}")
        total_train, total_val = train_pids.size, val_pids.size
        train_pids = MultimodalUKBReader.filter_participants_with_expansion_packs(
            train_pids, expansion_packs=cfg.expansion_packs, any=True
        )
        val_pids = MultimodalUKBReader.filter_participants_with_expansion_packs(
            val_pids, expansion_packs=cfg.expansion_packs, any=True
        )
        print(f"{train_pids.size} / {total_train} train pids")
        print(f"{val_pids.size} / {total_val} val pids")

    reader = MultimodalUKBReader(
        biomarkers=cfg.biomarkers, expansion_packs=cfg.expansion_packs
    )

    # vocab size + ignore_tokens are resolved up front: the tiebreak transform
    # consumes the final ignore_tokens, so this must run before TokenTransform.
    cfg.model.vocab_size = reader.vocab_size
    if cfg.ignore_expansion_packs:
        cfg.model.ignore_tokens = list(
            set(cfg.model.ignore_tokens).union(reader.expansion_tokens)
        )

    tiebreak_kwargs: dict = {}
    dx_token = None
    if cfg.tiebreak:
        # dx anchor takes the next free id; widen the vocab by one for it, and exempt
        # it from self-termination so the model can re-emit it once per cluster.
        dx_token = reader.vocab_size
        cfg.model.vocab_size = dx_token + 1
        cfg.model.self_terminate_except = list(
            set(cfg.model.self_terminate_except).union({dx_token})
        )
        # dx is a valid target/generatable token but not a disease -> exclude it
        # from disease eval via augmentation_tokens (default already holds no_event).
        cfg.model.augmentation_tokens = list(
            set(cfg.model.augmentation_tokens).union({dx_token})
        )
        whitelist_tokens = [0, 1, dx_token] + [
            reader.tokenizer[k] for k in reader.sex_keys + reader.lifestyle_keys
        ]
        tiebreak_kwargs = dict(
            break_clusters=True,
            dx_token=dx_token,
            whitelist_tokens=whitelist_tokens,
            death_token=reader.tokenizer["death"],
            ignore_tokens=cfg.model.ignore_tokens,
        )

    if cfg.exclude_smoking_and_alcohol:
        blacklist_tokens = [
            reader.tokenizer[k] for k in reader.smoking_keys + reader.alcohol_keys
        ]
    else:
        blacklist_tokens = None
    token_transform = TokenTransform(
        block_size=cfg.model.block_size,
        blacklist_tokens=blacklist_tokens,
        seed=cfg.seed,
        **tiebreak_kwargs,
    )

    if cfg.biomarkers is not None:
        if cfg.z_score_biomarkers:
            mean_dict = dict()
            std_dict = dict()
            for biomarker in cfg.biomarkers:
                mu, sigma = reader.biomarkers[biomarker].stats(train_pids)
                mean_dict[biomarker] = mu
                std_dict[biomarker] = sigma
        else:
            mean_dict = None
            std_dict = None
        train_biomarker_transform = BiomarkerTransform(
            biomarker2idx=reader.biomarker2idx,
            first_time_only=cfg.first_time_only,
            dropout=cfg.biomarker_dropout,
            seed=cfg.seed,
            z_score=cfg.z_score_biomarkers,
            mean=mean_dict,
            std=std_dict,
        )
        val_biomarker_transform = train_biomarker_transform.replace(dropout=None)
    else:
        train_biomarker_transform = None
        val_biomarker_transform = None

    train_ds = MultimodalDataset(
        reader=reader,
        pids=train_pids,
        token_transform=token_transform,
        biomarker_transform=train_biomarker_transform,
    )
    val_ds = MultimodalDataset(
        reader=reader,
        pids=val_pids,
        token_transform=token_transform,
        biomarker_transform=val_biomarker_transform,
    )

    if cfg.biomarkers is not None:
        for biomarker in cfg.biomarkers:
            projector = "linear"
            if biomarker in {"nmr", "proteomics"}:
                projector = "mlp"
            cfg.model.biomarkers[biomarker] = {
                "projector": projector,
                "input_size": Biomarker.input_size(biomarker),
            }
        cfg.model.biomarker2idx = reader.biomarker2idx
    model = DelphiM4(cfg.model)

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
        "reader_args": {
            "biomarkers": cfg.biomarkers,
            "expansion_packs": cfg.expansion_packs,
        },
        "tokenizer": (
            {**train_ds.tokenizer, "dx": dx_token}
            if cfg.tiebreak
            else train_ds.tokenizer
        ),
        **token_transform.to_ckpt(),
    }
    if train_biomarker_transform is not None:
        metadata |= train_biomarker_transform.to_ckpt()
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
