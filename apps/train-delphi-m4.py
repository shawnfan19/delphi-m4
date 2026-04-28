from dataclasses import asdict, dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from delphi import distributed
from delphi.data import MultimodalDataset
from delphi.data.transform import BiomarkerTransform, TokenTransform
from delphi.data.ukb import (
    Biomarker,
    MultimodalUKBReader,
    filter_participants_with_biomarkers,
    filter_participants_with_expansion_packs,
)
from delphi.experiment import BaseTrainer, Logger, TrainBaseConfig, seed_everything
from delphi.log import Checkpointer
from delphi.model.multimodal import DelphiM4, DelphiM4Config
from delphi.multimodal import Modality, parse_panel


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
    panel: None | str = None
    biomarkers: None | list[str] = None
    first_time_only: bool = False
    must_have: bool = False
    z_score_biomarkers: bool = True
    expansion_packs: None | list[str] = None
    must_have_expansion_packs: bool = False
    ignore_expansion_packs: bool = True
    biomarker_dropout: None | float = None

    def __post_init__(self):
        if self.panel:
            self.biomarkers, self.expansion_packs, self.panel_name = parse_panel(
                self.panel
            )


def train(cfg: TrainConfig):

    seed_everything(cfg.seed)

    train_pids = MultimodalUKBReader.participants(cfg.train_fold)
    val_pids = MultimodalUKBReader.participants(cfg.val_fold)
    if cfg.must_have:
        print(f"keeping participants with any of: {cfg.biomarkers}")
        total_train, total_val = train_pids.size, val_pids.size
        train_pids = filter_participants_with_biomarkers(
            train_pids, biomarkers=cfg.biomarkers, any=True
        )
        val_pids = filter_participants_with_biomarkers(
            val_pids, biomarkers=cfg.biomarkers, any=True
        )
        print(f"{train_pids.size} / {total_train} train pids")
        print(f"{val_pids.size} / {total_val} val pids")

    if cfg.must_have_expansion_packs:
        print(f"keeping participants with any of: {cfg.expansion_packs}")
        total_train, total_val = train_pids.size, val_pids.size
        train_pids = filter_participants_with_expansion_packs(
            train_pids, expansion_packs=cfg.expansion_packs, any=True
        )
        val_pids = filter_participants_with_expansion_packs(
            val_pids, expansion_packs=cfg.expansion_packs, any=True
        )
        print(f"{train_pids.size} / {total_train} train pids")
        print(f"{val_pids.size} / {total_val} val pids")

    reader = MultimodalUKBReader(
        biomarkers=cfg.biomarkers, expansion_packs=cfg.expansion_packs
    )
    token_transform = TokenTransform(block_size=cfg.model.block_size, seed=cfg.seed)

    if cfg.biomarkers is not None:
        if cfg.z_score_biomarkers:
            mean_dict = dict()
            std_dict = dict()
            for biomarker in cfg.biomarkers:
                biomarker = Modality[biomarker.upper()]
                mu, sigma = reader.biomarkers[biomarker].stats(train_pids)
                mean_dict[biomarker] = mu
                std_dict[biomarker] = sigma
        else:
            mean_dict = None
            std_dict = None
        biomarker_transform = BiomarkerTransform(
            first_time_only=cfg.first_time_only,
            seed=cfg.seed,
            z_score=cfg.z_score_biomarkers,
            mean=mean_dict,
            std=std_dict,
        )
    else:
        biomarker_transform = None

    train_ds = MultimodalDataset(
        reader=reader,
        pids=train_pids,
        token_transform=token_transform,
        biomarker_transform=biomarker_transform,
    )
    val_ds = MultimodalDataset(
        reader=reader,
        pids=val_pids,
        token_transform=token_transform,
        biomarker_transform=biomarker_transform,
    )

    cfg.model.vocab_size = reader.vocab_size
    if cfg.ignore_expansion_packs:
        cfg.model.ignore_tokens = list(
            set(cfg.model.ignore_tokens).union(reader.expansion_tokens)
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
        "token_transform_args": token_transform.config,
        "tokenizer": train_ds.tokenizer,
    }
    if biomarker_transform is not None:
        metadata["biomarker_transform_args"] = biomarker_transform.config
        metadata["biomarker_stats"] = biomarker_transform.stats
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
