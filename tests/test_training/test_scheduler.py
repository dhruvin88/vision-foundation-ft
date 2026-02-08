"""Tests for learning rate schedulers."""

import math

import pytest
import torch

from core.training.scheduler import (
    WarmupCosineScheduler,
    WarmupStepScheduler,
    build_scheduler,
)


@pytest.fixture
def optimizer():
    """Create a simple optimizer for scheduler testing."""
    model = torch.nn.Linear(10, 2)
    return torch.optim.SGD(model.parameters(), lr=0.1)


class TestWarmupCosineScheduler:
    """Tests for WarmupCosineScheduler."""

    def test_warmup_phase(self, optimizer):
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=5, total_epochs=50
        )
        # At epoch 0, LR should be 1/5 of base LR
        lrs = scheduler.get_lr()
        assert len(lrs) == 1
        assert lrs[0] == pytest.approx(0.1 * (1 / 5), rel=1e-5)

    def test_warmup_increases_lr(self, optimizer):
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=5, total_epochs=50
        )
        lrs = []
        for _ in range(5):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
        # LR should increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1]

    def test_cosine_decay_after_warmup(self, optimizer):
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=5, total_epochs=50
        )
        # Step through warmup
        for _ in range(5):
            scheduler.step()
        # Now in cosine phase, LR should decrease
        lr_at_warmup_end = scheduler.get_last_lr()[0]
        scheduler.step()
        lr_after = scheduler.get_last_lr()[0]
        assert lr_after < lr_at_warmup_end

    def test_min_lr_respected(self, optimizer):
        min_lr = 1e-6
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=2, total_epochs=10, min_lr=min_lr
        )
        for _ in range(100):
            scheduler.step()
        # LR should not go below min_lr
        assert scheduler.get_last_lr()[0] >= min_lr


class TestWarmupStepScheduler:
    """Tests for WarmupStepScheduler."""

    def test_warmup_phase(self, optimizer):
        scheduler = WarmupStepScheduler(
            optimizer, warmup_epochs=5, step_size=10, gamma=0.1
        )
        lrs = scheduler.get_lr()
        assert lrs[0] == pytest.approx(0.1 * (1 / 5), rel=1e-5)

    def test_step_decay_after_warmup(self, optimizer):
        scheduler = WarmupStepScheduler(
            optimizer, warmup_epochs=2, step_size=5, gamma=0.5
        )
        # Step through warmup
        for _ in range(2):
            scheduler.step()
        lr_after_warmup = scheduler.get_last_lr()[0]
        # Step through one step_size period
        for _ in range(5):
            scheduler.step()
        lr_after_step = scheduler.get_last_lr()[0]
        assert lr_after_step == pytest.approx(lr_after_warmup * 0.5, rel=1e-5)

    def test_multiple_step_decays(self, optimizer):
        scheduler = WarmupStepScheduler(
            optimizer, warmup_epochs=0, step_size=5, gamma=0.1
        )
        # After 10 epochs with step_size=5, should decay twice
        for _ in range(10):
            scheduler.step()
        expected = 0.1 * (0.1 ** 2)
        assert scheduler.get_last_lr()[0] == pytest.approx(expected, rel=1e-5)


class TestBuildScheduler:
    """Tests for the build_scheduler factory function."""

    def test_build_cosine(self, optimizer):
        scheduler = build_scheduler(optimizer, scheduler_type="cosine", total_epochs=50)
        assert isinstance(scheduler, WarmupCosineScheduler)

    def test_build_step(self, optimizer):
        scheduler = build_scheduler(optimizer, scheduler_type="step", total_epochs=50)
        assert isinstance(scheduler, WarmupStepScheduler)

    def test_build_constant(self, optimizer):
        scheduler = build_scheduler(optimizer, scheduler_type="constant")
        assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)

    def test_build_unknown_raises(self, optimizer):
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            build_scheduler(optimizer, scheduler_type="invalid")
