from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import torch
from torch.types import Tensor

from .dataset_wrapper import DatasetWrapper

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.configs.datasets import HILPDatasetConfig


class HILPDataset(DatasetWrapper):
    """Dataset wrapper used by the HILP objective."""

    identifier = 'hilp'
    aliases = ['hilp_dataset', 'HILPDataset', 'HILP']
    

    def __init__(self, cfg: HILPDatasetConfig) -> None:
        super().__init__(cfg)
        self.future_p = cfg.future_p
        self.gamma = cfg.gamma


    def _sample_goal(self, current_ep_idx: int, current_t: int) -> NDArray:
        """Samples a goal for given episode and timestep indices."""
        # Samples from future of current trajectory
        if np.random.random() < self.future_p:
            max_t = self.episode_lengths[current_ep_idx]

            # Checks if the current timestep is the last timestep, returns final state
            if current_t >= max_t:
                return self.get_obs(current_ep_idx, current_t)
            
            # Samples from Geom(1 - gamma)
            delta_t = np.random.geometric(1 - self.gamma)

            # Clips goal t to the end of trajectory
            goal_t = min(current_t + delta_t, max_t)

            return self.get_obs(current_ep_idx, goal_t)
        # Samples uniformly from all other states
        else:
            idx = np.random.randint(len(self.indices))
            i, t = self.indices[idx]
            return self.get_obs(i, t)


    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        i, t = self.indices[idx]

        state = self.get_obs(i, t)
        next_state = self.get_obs(i, t + 1)
        goal = self._sample_goal(i, t)

        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(next_state).float(),
            torch.from_numpy(goal).float()
        )
