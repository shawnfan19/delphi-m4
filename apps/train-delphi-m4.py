from dataclasses import dataclass, field

from omegaconf import OmegaConf

from delphi import distributed
from delphi.data.ukb import MultimodalUKBDataset
from delphi.experiment import BaseTrainer, TrainBaseConfig
from delphi.log import TrainLogConfig
from delphi.model.multimodal import DelphiM4, DelphiM4Config


@dataclass
class TrainConfig(TrainBaseConfig):
    ckpt_dir: str = "delphi-m4"
    batch_size: int = 128
    train_subject_list: str = "participants/train_fold.bin"
    val_subject_list: str = "participants/val_fold.bin"
    model: DelphiM4Config = field(
        default_factory=lambda: DelphiM4Config(block_size=256)
    )
    biomarkers: None | dict[str, int] = None
    z_score_biomarkers: bool = False
    first_time_only: bool = False
    must_have: bool = False
    expansion_packs: None | list[str] = None
    log: TrainLogConfig = field(default_factory=lambda: TrainLogConfig())


def train(cfg: TrainConfig):

    biomarkers = list(cfg.biomarkers.keys()) if cfg.biomarkers is not None else None
    data_args = {
        "expansion_packs": cfg.expansion_packs,
        "block_size": cfg.model.block_size,
        "first_time_only": cfg.first_time_only,
    }
    if cfg.must_have:
        data_args["must_have_biomarkers"] = biomarkers
    train_ds = MultimodalUKBDataset(
        subject_list=cfg.train_subject_list,
        biomarkers=biomarkers,
        z_score_biomarkers=cfg.z_score_biomarkers,
        **data_args,
    )
    val_ds = MultimodalUKBDataset(
        subject_list=cfg.val_subject_list,
        biomarker_datasets=train_ds.mod_ds,
        perturb=False,
        **data_args,
    )

    cfg.model.vocab_size = train_ds.vocab_size
    cfg.model.ignore_tokens = list(
        set(cfg.model.ignore_tokens).union(train_ds.expansion_tokens)
    )
    if cfg.biomarkers is not None:
        for biomarker, n_features in cfg.biomarkers.items():
            cfg.model.biomarkers[biomarker] = {
                "projector": "linear",
                "input_size": n_features,
            }
    model = DelphiM4(cfg.model)

    backend = distributed.make_backend_from_args(cfg)
    cfg.log.wandb_project = cfg.ckpt_dir
    trainer = BaseTrainer(
        cfg=cfg,
        backend=backend,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
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
