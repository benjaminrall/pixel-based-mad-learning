from __future__ import annotations

import numpy as np
import torch

from .dataset_wrapper import DatasetWrapper

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.configs.datasets import MadDistDatasetConfig


class MadDistDataset(DatasetWrapper):
    """Dataset wrapper used by the MadDist objective."""

    identifier = 'maddist'
    aliases = ['maddist_dataset', 'MadDistDataset', 'MadDist', 'mad', 'MAD']
    
    def __init__(self, cfg: MadDistDatasetConfig) -> None:
        super().__init__(cfg)
        self.constraint_batch_size = cfg.constraint_batch_size
        self.H_c = cfg.H_c


    def __getitem__(self, idx: int) -> tuple:
        # Gets state_i
        i, t = self.indices[idx]
        state_i = self.get_obs(i, t)

        max_t = self.episode_lengths[i]

        # Gets state_j from the same trajectory
        j = np.random.randint(t + 1, max_t + 1)
        state_j = self.get_obs(i, j)

        diff = j - t

        # Random state sampling for contrastive loss
        rand_idx = np.random.randint(len(self.indices))
        rand_i, rand_t = self.indices[rand_idx]
        state_r = self.get_obs(rand_i, rand_t)

        return (
            # s_i, s_j, j - i, s_r
            torch.from_numpy(state_i).float(),
            torch.from_numpy(state_j).float(),
            torch.tensor([diff], dtype=torch.float32),
            torch.from_numpy(state_r).float(),
        )
