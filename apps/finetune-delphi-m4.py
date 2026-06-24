from dataclasses import asdict, dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path

import torch

from delphi import distributed
from delphi.data import MultimodalDataset
from delphi.data.auto import multimodal_reader_cls
from delphi.data.transform import BiomarkerTransform, TokenTransform
from delphi.env import DELPHI_CKPT_READ as DELPHI_CKPT_DIR
from delphi.experiment import BaseTrainer, Logger, TrainBaseConfig, seed_everything
from delphi.log import Checkpointer
from delphi.model.multimodal import DelphiM4, DelphiM4Config
from delphi.multimodal import Modality
from delphi.optim import (
    configure_optimizer,
    merge_param_groups,
    parse_differential_lr_groups,
    parse_weight_decay_groups,
)


@dataclass
class FinetuneConfig(TrainBaseConfig):
    pretrain_ckpt: str = ""
    baseline_only: bool = False
    batch_size: int = 128
    seed: int = 42
    deterministic: bool = False
    model: DelphiM4Config = field(
        default_factory=lambda: DelphiM4Config(block_size=None)
    )
    biomarkers: list[str] = field(default_factory=list)
    first_time_only: bool = True
    z_score_biomarkers: bool = True
    biomarker_dropout: None | float = None
    freeze_backbone: bool = False

    learning_rate: float = 1e-5
    schedule: str = "cosine"
    max_iters: int = 20000
    warmup_iters: float | int = 0.1

    differential_lr: float = 1e-3
    eval_interval: int = 200


def finetune(cfg: FinetuneConfig):

    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    # dataset-aware: UKB on the cluster, AoU on the workbench. Honors
    # DELPHI_DATASET (set in the dsub env), else auto-detects from the data dir.
    ReaderCls = multimodal_reader_cls()
    Biomarker = ReaderCls.biomarker_cls

    # Load pre-trained checkpoint
    assert cfg.pretrain_ckpt, "pretrained_ckpt must be specified"
    ckpt_dict = torch.load(
        Path(DELPHI_CKPT_DIR) / cfg.pretrain_ckpt, map_location="cpu"
    )
    pretrained_model_args = ckpt_dict["model_args"]
    pretrain_cfg = ckpt_dict["config"]
    pretrain_biomarkers = list(pretrained_model_args.get("biomarkers", {}).keys())

    assert len(cfg.biomarkers) > 0, "new_biomarkers must not be empty"
    overlap = set(cfg.biomarkers) & set(pretrain_biomarkers)
    assert len(overlap) == 0, f"new_biomarkers overlap with pretrained: {overlap}"
    print(f"pre-trained biomarkers: {pretrain_biomarkers }")
    print(f"new biomarkers: {cfg.biomarkers}")

    # Build model config from checkpoint
    valid_fields = {f.name for f in dc_fields(DelphiM4Config)}
    model_args = {k: v for k, v in pretrained_model_args.items() if k in valid_fields}
    model_cfg = DelphiM4Config(**model_args)

    if not cfg.baseline_only:
        all_biomarkers = pretrain_biomarkers + cfg.biomarkers
    else:
        all_biomarkers = pretrain_biomarkers

    train_pids = ReaderCls.participants("train")
    val_pids = ReaderCls.participants("val")
    print(f"keeping participants with any of: {cfg.biomarkers}")
    total_train, total_val = train_pids.size, val_pids.size
    train_pids = ReaderCls.filter_participants_with_biomarkers(
        train_pids, biomarkers=cfg.biomarkers, any=True
    )
    val_pids = ReaderCls.filter_participants_with_biomarkers(
        val_pids, biomarkers=cfg.biomarkers, any=True
    )
    print(f"{train_pids.size} / {total_train} train pids")
    print(f"{val_pids.size} / {total_val} val pids")

    reader = ReaderCls(
        biomarkers=all_biomarkers, expansion_packs=pretrain_cfg["expansion_packs"]
    )

    token_transform = TokenTransform(block_size=model_cfg.block_size, seed=cfg.seed)

    if cfg.z_score_biomarkers:
        mean_dict = dict()
        std_dict = dict()
        if pretrain_biomarkers:
            pretrain_stats = ckpt_dict.get("biomarker_stats") or {}
            pretrain_mean = pretrain_stats.get("mean") or {}
            pretrain_std = pretrain_stats.get("std") or {}
            missing = set(pretrain_biomarkers) - set(pretrain_mean)
            assert not missing, (
                f"pretrained checkpoint missing biomarker stats for {missing}; "
                "cannot z-score without breaking the pretrained input distribution"
            )
            mean_dict.update(
                {Modality[k.upper()]: pretrain_mean[k] for k in pretrain_biomarkers}
            )
            std_dict.update(
                {Modality[k.upper()]: pretrain_std[k] for k in pretrain_biomarkers}
            )
        if not cfg.baseline_only:
            for biomarker in cfg.biomarkers:
                modality = Modality[biomarker.upper()]
                mu, sigma = reader.biomarkers[modality].stats(train_pids)
                mean_dict[modality] = mu
                std_dict[modality] = sigma
    else:
        mean_dict = None
        std_dict = None

    train_bio_transform = BiomarkerTransform(
        biomarker2idx=reader.biomarker2idx,
        dropout=cfg.biomarker_dropout,
        z_score=cfg.z_score_biomarkers,
        mean=mean_dict,
        std=std_dict,
        first_time_only=cfg.first_time_only,
        seed=cfg.seed,
    )
    train_bio_transform.describe()
    val_bio_transform = BiomarkerTransform(
        biomarker2idx=reader.biomarker2idx,
        dropout=False,
        z_score=cfg.z_score_biomarkers,
        mean=mean_dict,
        std=std_dict,
        first_time_only=cfg.first_time_only,
        seed=cfg.seed,
    )

    train_ds = MultimodalDataset(
        reader=reader,
        pids=train_pids,
        token_transform=token_transform,
        biomarker_transform=train_bio_transform,
    )
    val_ds = MultimodalDataset(
        reader=reader,
        pids=val_pids,
        token_transform=token_transform,
        biomarker_transform=val_bio_transform,
    )

    if not cfg.baseline_only:
        for biomarker in cfg.biomarkers:
            projector = "linear"
            # if biomarker in {"nmr", "proteomics"}:
            #     projector = "mlp"
            model_cfg.biomarkers[biomarker] = {
                "projector": projector,
                "input_size": Biomarker.input_size(biomarker),
            }
        # ensure modality embedding is enabled for finetuning even if the
        # pretrained model didn't have one (e.g. no biomarkers at all)
        model_cfg.modality_emb = True
    cfg.model = model_cfg

    # Create model with extended config
    model = DelphiM4(model_cfg)
    pretrained_state = ckpt_dict["model"]

    # Handle mod_embedding: copy pretrained rows if they exist, otherwise
    # the entire embedding is new and will be learned from scratch
    old_mod_idx = [
        Modality[biomarker.upper()].value for biomarker in pretrain_biomarkers
    ]
    mod_emb_key = "transformer.embed.mod_embedding.weight"
    if mod_emb_key in pretrained_state:
        old_mod_weight = pretrained_state.pop(mod_emb_key)
        preserve_idx = [0] + old_mod_idx
        model.transformer.embed.mod_embedding.weight.data[preserve_idx] = (
            old_mod_weight[preserve_idx]
        )

    new, _ = model.load_state_dict(pretrained_state, strict=False)
    print(f"new modules: {new}")

    # Freeze all parameters
    if cfg.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        model.transformer.embed.unfreeze_for_new_biomarkers(biomarkers=cfg.biomarkers)

        model.transformer.eval()
        for biomarker in cfg.biomarkers:
            bm_key = Modality[biomarker.upper()].name.lower()
            model.transformer.embed.biomarker_embed[bm_key].train()

        optimizer = None
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        lr_param_groups = parse_differential_lr_groups(
            model=model, new_keys=new, differential_lr=cfg.differential_lr
        )
        wd_param_groups = parse_weight_decay_groups(model=model)
        param_groups = merge_param_groups(
            lr_param_groups=lr_param_groups, wd_param_groups=wd_param_groups
        )
        optimizer = configure_optimizer(
            optim_groups=param_groups,
            learning_rate=cfg.learning_rate,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
        )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"trainable parameters: {n_trainable:,} / {n_total:,} "
        f"({n_trainable / n_total:.2%})"
    )

    # Train
    backend = distributed.make_backend_from_args(cfg)
    cfg.wandb_project = cfg.ckpt_dir

    n_params = sum(p.numel() for p in model.parameters())
    logger = Logger(
        config=asdict(cfg),
        backend=backend,
        wandb_log=cfg.wandb_log,
        wandb_project=cfg.wandb_project,
        tensorboard_log=cfg.tensorboard_log,
        tensorboard_dir=cfg.tensorboard_dir,
        run_name=cfg.run_name,
        summary={"model_params": n_params},
    )

    metadata = {
        "config": asdict(cfg),
        "model_args": asdict(cfg.model),
        "pretrain_ckpt": cfg.pretrain_ckpt,
        "reader_args": {
            "biomarkers": all_biomarkers,
            "expansion_packs": pretrain_cfg["expansion_packs"],
        },
        "tokenizer": train_ds.tokenizer,
        **token_transform.to_ckpt(),
        **val_bio_transform.to_ckpt(),
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
        optimizer=optimizer,
    )
    trainer.train()
    backend.finalize()


def main():
    finetune(FinetuneConfig.from_cli())


if __name__ == "__main__":
    main()
