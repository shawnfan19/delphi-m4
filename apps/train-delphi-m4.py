from dataclasses import dataclass, field

from omegaconf import OmegaConf

from delphi import distributed
from delphi.data.ukb import MultimodalUKBDataset
from delphi.experiment import BaseTrainer, TrainBaseConfig, seed_everything
from delphi.log import TrainLogConfig
from delphi.model.multimodal import DelphiM4, DelphiM4Config
from delphi.multimodal import module_name


@dataclass
class TrainConfig(TrainBaseConfig):
    ckpt_dir: str = "debug"
    batch_size: int = 128
    seed: int = 42
    deterministic: bool = False
    train_subject_list: str = "participants/train_fold.bin"
    val_subject_list: str = "participants/val_fold.bin"
    model: DelphiM4Config = field(
        default_factory=lambda: DelphiM4Config(block_size=None)
    )
    biomarkers: None | list[str] = None
    first_time_only: bool = False
    must_have: bool = False
    z_score_biomarkers: bool = True
    expansion_packs: None | list[str] = None
    biomarker_dropout: None | float = None
    log: TrainLogConfig = field(default_factory=lambda: TrainLogConfig())


def train(cfg: TrainConfig):

    seed_everything(cfg.seed)

    biomarkers = cfg.biomarkers
    data_args = {
        "expansion_packs": cfg.expansion_packs,
        "block_size": cfg.model.block_size,
        "first_time_only": cfg.first_time_only,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "z_score_biomarkers": cfg.z_score_biomarkers,
    }
    if cfg.must_have:
        data_args["must_have_biomarkers"] = biomarkers
    train_ds = MultimodalUKBDataset(
        subject_list=cfg.train_subject_list,
        biomarkers=biomarkers,
        biomarker_dropout=cfg.biomarker_dropout,
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
        for modality, ds in train_ds.mod_ds.items():
            biomarker = module_name(modality)
            projector = "linear"
            if biomarker == "nmr":
                projector = "mlp"
            cfg.model.biomarkers[biomarker] = {
                "projector": projector,
                "input_size": ds.n_features,
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
