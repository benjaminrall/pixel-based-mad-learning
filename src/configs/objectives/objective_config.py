from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TypeVar

from src.configs import Config
from src.configs.datasets import DatasetConfig

from src.datasets import DatasetWrapper
from src.utils import Registrable
from src.objectives import Objective

O = TypeVar('O', bound='Objective', covariant=True)


@dataclass
class ObjectiveConfig(Config[O], Registrable['ObjectiveConfig'], ABC):
    """Abstract base class for config classes for all trainable objectives."""

    dataset: Config[DatasetWrapper]
    total_updates: int = 1000000

    def __post_init__(self) -> None:
        """Automatically converts a dataset dictionary to a DatasetConfig instance."""
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig.from_dict(self.dataset)
