from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TypeVar

from src.configs import Config
from src.utils import Registrable
from src.datasets import DatasetWrapper

D = TypeVar('D', bound='DatasetWrapper', covariant=True)


@dataclass
class DatasetConfig(Config[D], Registrable['DatasetConfig'], ABC):
    """Abstract base class for config classes for all Minari dataset wrappers."""

    dataset_id: str = 'D4RL/pointmaze/umaze-v2'
    obs_keys: list[str] = field(default_factory=lambda : ['observation'])
    info_keys: dict[str, str] = field(default_factory=dict)

    batch_size: int = 1024
    shuffle: bool = True
    num_workers: int = 1
    drop_last: bool = True
    mmap_obs: bool = True
