from __future__ import annotations

from abc import ABC, abstractmethod

from src.utils import Registrable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src import Trainer
    from src.objectives import Objective
    from src.configs.callbacks import CallbackConfig


class Callback(Registrable['Callback'], ABC):
    """Abstract base class for all trainer callbacks."""

    def __init__(self, cfg: CallbackConfig) -> None:
        super().__init__()
        self.cfg = cfg


    @abstractmethod
    def on_train_start(self, trainer: Trainer, objective: Objective) -> None:
        """Callback method called once at the beginning of a training run."""


    @abstractmethod
    def on_update(self, trainer: Trainer, objective: Objective, progress: int) -> None:
        """Callback method called at the end of each training update."""


    @abstractmethod
    def on_train_end(self, trainer: Trainer, objective: Objective) -> None:
        """Callback method called once at the end of a training run."""
