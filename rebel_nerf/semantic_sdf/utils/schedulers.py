from dataclasses import dataclass, field
from typing import Type

from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.engine.schedulers import Scheduler, SchedulerConfig
from torch.optim import lr_scheduler

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


@dataclass
class ExponentialSchedulerConfig(SchedulerConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: ExponentialScheduler)
    decay_rate: float = 0.1
    max_steps: int = 1000000


class ExponentialScheduler(Scheduler):
    config: ExponentialSchedulerConfig

    def get_scheduler(self, optimizer: Optimizers, lr_init: float) -> LRScheduler:
        return lr_scheduler.ExponentialLR(
            optimizer,
            self.config.decay_rate ** (1.0 / self.config.max_steps),
        )
