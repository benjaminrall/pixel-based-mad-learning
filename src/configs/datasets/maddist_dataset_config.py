from __future__ import annotations

from dataclasses import dataclass

from .dataset_config import DatasetConfig

from src.datasets import MadDistDataset


@dataclass
class MadDistDatasetConfig(DatasetConfig['MadDistDataset']):

    constraint_batch_size: int = 1024
    H_c: int = 6

    identifier = 'maddist'
    aliases = ['maddist_dataset', 'MadDistDataset', 'MadDist', 'mad' 'MAD']

    @property
    def target_class(self) -> type[MadDistDataset]:
        return MadDistDataset
