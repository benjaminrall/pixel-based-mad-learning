from __future__ import annotations

from dataclasses import dataclass

from .dataset_config import DatasetConfig

from src.datasets import HILPDataset


@dataclass
class HILPDatasetConfig(DatasetConfig['HILPDataset']):

    future_p: float = 0.625
    gamma: float = 0.99

    identifier = 'hilp'
    aliases = ['hilp_dataset', 'HILPDataset', 'HILP']

    @property
    def target_class(self) -> type[HILPDataset]:
        return HILPDataset
