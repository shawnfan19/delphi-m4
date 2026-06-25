"""Temporal point process likelihood layer for DelphiM4.

Public surface is re-exported here so ``from delphi.model.tpp import X`` keeps
working after the split into per-loss-family submodules:

- ``homo_poisson`` — :class:`HomoPoissonTPP`
- ``neural`` — neural-intensity family (:class:`NeuralIntensity`,
  :class:`NeuralTPP`, :class:`NeuralODEIntensity`, :class:`NeuralODETPP`)
- ``sets`` — set-valued dynamic DPP (:class:`DPPSetHead`,
  :class:`DynamicDPPTPP`)
- ``dispatch`` — cross-cutting glue (:func:`tpp_dispatch`,
  :func:`conditional_log_likelihood`)
"""

from .dispatch import conditional_log_likelihood, tpp_dispatch
from .homo_poisson import HomoPoissonTPP
from .neural import NeuralIntensity, NeuralODEIntensity, NeuralODETPP, NeuralTPP
from .sets import DPPSetHead, DynamicDPPTPP

__all__ = [
    "HomoPoissonTPP",
    "NeuralIntensity",
    "NeuralTPP",
    "NeuralODEIntensity",
    "NeuralODETPP",
    "DPPSetHead",
    "DynamicDPPTPP",
    "tpp_dispatch",
    "conditional_log_likelihood",
]
