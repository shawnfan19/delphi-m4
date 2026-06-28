"""Logger selects one backend by name: an unknown log_backend is rejected, and
a non-master backend builds no backend instance -- so this runs without
wandb/tensorboard/trackio installed."""

from typing import cast

import pytest

from delphi import distributed
from delphi.log import Logger


class _NonMaster:
    """Minimal DistributedBackend stand-in. Non-master => Logger builds no
    backend instance, exercising selection/validation without importing any."""

    def is_master_process(self):
        return False


# Logger only calls backend.is_master_process(); cast keeps the type checker happy.
_NM = cast(distributed.backend.DistributedBackend, _NonMaster())


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        Logger(config={}, backend=_NM, log_backend="weights_and_biases")


def test_valid_backends_ok():
    # non-master never constructs a backend, so .log_backend stays None for each
    for lb in ("wandb", "tensorboard", "trackio", "none"):
        assert Logger(config={}, backend=_NM, log_backend=lb).log_backend is None


def test_trackio_backend_strips_step():
    """TrackioBackend.log drops the redundant in-dict 'step' (carried by step=,
    and a reserved key in trackio) but forwards everything else + step=."""
    from delphi.log import TrackioBackend

    class _FakeTrackio:
        def __init__(self):
            self.calls = []

        def log(self, metrics, step):
            self.calls.append((metrics, step))

    # bypass __init__ (it imports the real trackio); only exercise log()
    b = TrackioBackend.__new__(TrackioBackend)
    b.trackio = _FakeTrackio()  # type: ignore[assignment]
    b.log({"step": 5, "loss": 1.0}, step=5, commit=True)
    assert b.trackio.calls == [({"loss": 1.0}, 5)]


def test_config_validates_log_backend():
    """TrainBaseConfig.__post_init__ fails fast on a bad log_backend (from any
    source) before the reader/model build; valid values pass."""
    from delphi.experiment import TrainBaseConfig

    for lb in ("wandb", "tensorboard", "trackio", "none"):
        assert TrainBaseConfig(log_backend=lb).log_backend == lb
    with pytest.raises(ValueError):
        TrainBaseConfig(log_backend="trackiio")


if __name__ == "__main__":
    test_invalid_backend_raises()
    test_valid_backends_ok()
    test_trackio_backend_strips_step()
    test_config_validates_log_backend()
    print("OK")
