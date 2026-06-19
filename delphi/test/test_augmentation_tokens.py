"""augmentation_tokens: the model targets that are NOT meaningful diseases
(no_event, and the dx cluster anchor on tiebreak checkpoints).

Pins the contract eval scripts rely on: ``model.targets`` is the loss-scored set;
excluding ``model.augmentation_tokens`` from it yields the disease set (death and
ordinary diagnoses stay). The exclusion is written explicitly at each eval site,
so these tests guard the accessor + the idiom + the checkpoint round-trip.
"""

from dataclasses import asdict, fields

import torch

from delphi.model.multimodal import DelphiM4, DelphiM4Config


def _model(**cfg):
    return DelphiM4(DelphiM4Config(vocab_size=20, n_layer=1, n_head=2, n_embd=8, **cfg))


def _disease_ids(model):
    t = model.targets
    return t[~torch.isin(t, model.augmentation_tokens)].tolist()


def test_default_augmentation_is_no_event():
    m = _model()
    assert m.config.augmentation_tokens == [1]
    assert m.augmentation_tokens.tolist() == [1]
    dis = _disease_ids(m)
    assert 1 not in dis  # no_event excluded
    assert 13 in dis  # an ordinary disease id (not in ignore_tokens default) kept


def test_tiebreak_excludes_dx_keeps_diseases():
    dx = 19  # vocab_size - 1, the way training assigns it
    dis = _disease_ids(_model(augmentation_tokens=[1, dx]))
    assert 1 not in dis and dx not in dis  # no_event + dx anchor both excluded
    assert 13 in dis  # a real disease (e.g. death in the full vocab) still scored


def test_augmentation_tokens_round_trip():
    # mirrors load_ckpt: asdict(model config) -> filter to valid fields -> rebuild
    cfg = DelphiM4Config(vocab_size=20, augmentation_tokens=[1, 19])
    valid = {f.name for f in fields(DelphiM4Config)}
    cfg2 = DelphiM4Config(**{k: v for k, v in asdict(cfg).items() if k in valid})
    assert cfg2.augmentation_tokens == [1, 19]
