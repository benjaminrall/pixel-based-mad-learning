import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.types import Tensor

from src.datasets import MadDistDataset

from typing import Generator


class MadDistConstraintDatasetView(Dataset):
    """A specialised view of a MadDist dataset, used to fetch separate constraint batches."""

    def __init__(self, base: MadDistDataset) -> None:
        super().__init__()
        self.base = base


    def __len__(self) -> int:
        return len(self.base)
    

    def loader(self) -> DataLoader:
        """Returns a pre-configured DataLoader for the dataset view."""
        return DataLoader(
            self,
            batch_size=self.base.constraint_batch_size,
            shuffle=self.base.shuffle,
            num_workers=self.base.num_workers,
            drop_last=self.base.drop_last
        )
    

    def infinite_iterator(self) -> Generator:
        """Returns an infinite iterator over batches of the dataset view."""
        dl = self.loader()
        while True:
            for batch in dl:
                yield batch


    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        # Gets state_i
        i, t = self.base.indices[idx]
        state_i = self.base.get_obs(i, t)

        max_t = self.base.episode_lengths[i]

        # Gets state_j from the same trajectory, constrained by H_c
        max_j = min(t + self.base.H_c, max_t)
        j = np.random.randint(t + 1, max_j + 1)
        state_j = self.base.get_obs(i, j)

        diff = j - t

        return (
            # s_i, s_j, j - i
            torch.from_numpy(state_i).float(),
            torch.from_numpy(state_j).float(),
            torch.tensor([diff], dtype=torch.float32),
        )
