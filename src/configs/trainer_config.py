from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from src.configs.objectives import ObjectiveConfig
from src.objectives import Objective
from src.configs.callbacks import CallbackConfig
from src.callbacks import Callback

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src import Trainer


@dataclass
class TrainerConfig(Config['Trainer']):
    """Config class for experiments run by the Trainer."""

    run_name: str
    objective: Config[Objective]
    callbacks: list[Config[Callback]] = field(default_factory=list)
    run_id: str = ''
    seed: int = 42
    save_checkpoints: bool = True
    checkpoint_interval: int = 1
    checkpoint_folder: str = './checkpoints'
    track_wandb: bool = True
    wandb_project: str = 'Year 4 Project'
    wandb_entity: str | None = None
    wandb_dir: str | None = './logs'

    def __post_init__(self) -> None:
        if isinstance(self.objective, dict):
            self.objective = ObjectiveConfig.from_dict(self.objective)
        if self.callbacks:
            self.callbacks = [
                CallbackConfig.from_dict(c) 
                for c in self.callbacks if isinstance(c, dict)
            ]

    @property
    def target_class(self) -> type[Trainer]:
        from src import Trainer
        return Trainer
