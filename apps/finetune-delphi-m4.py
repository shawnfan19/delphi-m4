from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path

import torch
from omegaconf import OmegaConf

from delphi import distributed
from delphi.data.ukb import MultimodalUKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import BaseTrainer, TrainBaseConfig, seed_everything
from delphi.log import TrainLogConfig
from delphi.model.multimodal import DelphiM4, DelphiM4Config
from delphi.multimodal import Modality, module_name
from delphi.optim import OptimConfig


def unfreeze_biomarker_projectors(model, biomarkers):

    # Unfreeze new biomarker projectors
    for biomarker in biomarkers:
        name = module_name(Modality[biomarker.upper()])
        for param in model.transformer.embed.biomarker_embed[name].parameters():
            param.requires_grad = True


@dataclass
class FinetuneConfig(TrainBaseConfig):
    pretrain_ckpt: str = ""
    batch_size: int = 128
    seed: int = 42
    deterministic: bool = False
    model: DelphiM4Config = field(
        default_factory=lambda: DelphiM4Config(block_size=None)
    )
    biomarkers: list[str] = field(default_factory=list)
    first_time_only: bool = False
    z_score_biomarkers: bool = True
    biomarker_dropout: None | float = None
    freeze_backbone: bool = False
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            learning_rate=1e-5, schedule="constant", max_iters=5000
        )
    )
    eval_interval: int = 200
    log: TrainLogConfig = field(default_factory=lambda: TrainLogConfig())


def finetune(cfg: FinetuneConfig):

    seed_everything(cfg.seed)

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

    # Build datasets with all biomarkers (pre-trained + new)
    all_biomarkers = pretrain_biomarkers + cfg.biomarkers
    data_args = {
        "expansion_packs": pretrain_cfg["expansion_packs"],
        "block_size": model_cfg.block_size,
        "first_time_only": cfg.first_time_only,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "must_have_biomarkers": cfg.biomarkers,
        "biomarker_require": "any",
        "z_score_biomarkers": cfg.z_score_biomarkers,
    }

    train_ds = MultimodalUKBDataset(
        subject_list=pretrain_cfg["train_subject_list"],
        biomarkers=all_biomarkers,
        biomarker_dropout=cfg.biomarker_dropout,
        **data_args,
    )
    val_ds = MultimodalUKBDataset(
        subject_list=pretrain_cfg["val_subject_list"],
        biomarker_datasets=train_ds.mod_ds,
        perturb=False,
        **data_args,
    )

    # Extend model config with new biomarkers
    for modality, ds in train_ds.mod_ds.items():
        biomarker = module_name(modality)
        if biomarker not in model_cfg.biomarkers:
            projector = "linear"
            if biomarker in {"nmr", "proteomics"}:
                projector = "mlp"
            model_cfg.biomarkers[biomarker] = {
                "projector": projector,
                "input_size": ds.n_features,
            }
    # Ensure modality embedding is enabled for finetuning even if the
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

    missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
    print(f"missing keys (new modules): {missing}")
    if unexpected:
        print(f"unexpected keys: {unexpected}")

    # Freeze all parameters
    if cfg.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        unfreeze_biomarker_projectors(model=model, biomarkers=cfg.biomarkers)

        # Unfreeze modality embedding; freeze row 0 (padding) and any pretrained rows
        model.transformer.embed.mod_embedding.weight.requires_grad = True
        freeze_idx = [0] + old_mod_idx

        def mod_emb_grad_hook(grad, freeze_idx=freeze_idx):
            grad = grad.clone()
            grad[freeze_idx] = 0
            return grad

        model.transformer.embed.mod_embedding.weight.register_hook(mod_emb_grad_hook)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"trainable parameters: {n_trainable:,} / {n_total:,} "
        f"({n_trainable / n_total:.2%})"
    )

    # Train
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

    default_cfg = OmegaConf.structured(FinetuneConfig())
    cli_args = OmegaConf.from_cli()

    if hasattr(cli_args, "config"):
        file_cfg = OmegaConf.load(cli_args.config)
        del cli_args.config
    else:
        file_cfg = default_cfg

    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    finetune(cfg)


if __name__ == "__main__":
    main()
