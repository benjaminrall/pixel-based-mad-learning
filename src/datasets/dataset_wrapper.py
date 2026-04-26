from __future__ import annotations

from abc import ABC, abstractmethod
import os

import numpy as np
from numpy.typing import NDArray

from torch.utils.data import Dataset, DataLoader
from torch.types import Tensor

import minari

from tqdm import tqdm

from src.utils import Registrable

from typing import Generator, TYPE_CHECKING
if TYPE_CHECKING:
    from src.configs.datasets import DatasetConfig


class DatasetWrapper(Dataset, Registrable['DatasetWrapper'], ABC):
    """Base class for all Minari dataset wrappers."""

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.obs_keys = cfg.obs_keys
        self.info_keys = cfg.info_keys

        # Loader settings
        self.batch_size = cfg.batch_size
        self.shuffle = cfg.shuffle
        self.num_workers = cfg.num_workers
        self.drop_last = cfg.drop_last

        self.cache_dir = os.path.join('.cache', 'datasets', cfg.dataset_id)

        # Initialises an indexed cache for the dataset if it does not already exist
        self._init_cache(cfg.dataset_id)
        
        # Loads dataset elements from cached .npy files
        self.observations = np.load(
            os.path.join(self.cache_dir, 'observations.npy'), 
            mmap_mode='r' if cfg.mmap_obs else None
        )
        self.indices = np.load(os.path.join(self.cache_dir, 'indices.npy'))
        self.episode_lengths = np.load(os.path.join(self.cache_dir, 'episode_lengths.npy'))
        self.episode_starts = np.load(os.path.join(self.cache_dir, 'episode_starts.npy'))
        self.infos = {}
        for key in self.info_keys:
            info_path = os.path.join(self.cache_dir, f'info_{key}.npy')
            self.infos[key] = np.load(info_path)


    def _init_cache(self, dataset_id: str) -> None:
        """Initialises a cache containing the indexed observation data and additional info required for our datasets."""
        obs_path = os.path.join(self.cache_dir, 'observations.npy')
        indices_path = os.path.join(self.cache_dir, 'indices.npy')
        lengths_path = os.path.join(self.cache_dir, 'episode_lengths.npy')
        starts_path = os.path.join(self.cache_dir, 'episode_starts.npy')
        infos_paths = [os.path.join(self.cache_dir, f'info_{key}.npy') for key in self.info_keys]

        # Checks if all required elements already exist
        if all([os.path.exists(obs_path), os.path.exists(indices_path), os.path.exists(lengths_path), os.path.exists(starts_path)]
               + [os.path.exists(infos_path) for infos_path in infos_paths]):
            return

        # Creates cache directory and loads the Minari dataset
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset = minari.load_dataset(dataset_id, download=True)

        total_observations = 0
        obs_shape = ()
        obs_dtype = None

        lengths = []
        starts = []
        indices = []

        infos = {key: [] for key in self.info_keys}
        
        # Initial pass through the dataset
        for i, episode in tqdm(enumerate(dataset), desc='Indexing dataset', total=len(dataset)):
            # Stores the shape and dtype of the environment's observations
            if obs_dtype is None:
                obs_data = episode.observations
                if isinstance(obs_data, dict):
                    parts = [obs_data[k] for k in self.obs_keys]
                    sample_obs = np.concatenate(parts, axis=-1)
                else:
                    sample_obs = obs_data
                
                obs_shape = sample_obs.shape[1:]
                obs_dtype = sample_obs.dtype


            # Gets the length of an episode's observations
            if isinstance(episode.observations, dict):
                ep_obs_len = len(episode.observations[self.obs_keys[0]])
            else:
                ep_obs_len = len(episode.observations)

            # Updates episode start and length arrays
            steps = len(episode.actions)
            starts.append(total_observations)
            lengths.append(steps)
            total_observations += ep_obs_len

            # Updates observation indices arrays
            ep_indices = np.empty((steps, 2), dtype=np.int32)
            ep_indices[:, 0] = i
            ep_indices[:, 1] = np.arange(steps)
            indices.append(ep_indices)

            # Stores all requested info keys
            for key, origin in self.info_keys.items():
                if episode.infos is not None and origin in episode.infos:
                    infos[key].append(episode.infos[origin])
                elif isinstance(episode.observations, dict) and origin in episode.observations:
                    infos[key].append(episode.observations[origin])
        
        # Saves initial pass data to files
        np.save(indices_path, np.concatenate(indices, axis=0))
        np.save(lengths_path, np.array(lengths, dtype=np.int32))
        np.save(starts_path, np.array(starts, dtype=np.int32))
        for key in self.info_keys:
            info_path = os.path.join(self.cache_dir, f'info_{key}.npy')
            np.save(info_path, np.concatenate(infos[key], axis=0))

        # Prepares to write observations to file using memory mapping (to handle large visual datasets)
        mmap_observations = np.lib.format.open_memmap(
            obs_path, 
            mode='w+', 
            dtype=obs_dtype, 
            shape=(total_observations, *obs_shape)
        )

        # Second pass specifically to cache observations from the dataset
        current_obs_idx = 0
        for i, episode in tqdm(enumerate(dataset), desc='Caching observations', total=len(dataset)):
            obs_data = episode.observations
            if isinstance(obs_data, dict):
                parts = [obs_data[k] for k in self.obs_keys]
                ep_obs = np.concat(parts, axis=-1)
            else:
                ep_obs = obs_data

            ep_obs_len = len(ep_obs)
            mmap_observations[current_obs_idx : current_obs_idx + ep_obs_len] = ep_obs
            current_obs_idx += ep_obs_len

        mmap_observations.flush()


    def __len__(self) -> int:
        return len(self.indices)


    def loader(self) -> DataLoader:
        """Returns a pre-configured DataLoader for the dataset."""
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )


    def infinite_iterator(self) -> Generator:
        """Returns an infinite iterator over batches of the dataset."""
        dl = self.loader()
        while True:
            for batch in dl:
                yield batch


    def get_ep_obs(self, ep_idx: int) -> NDArray:
        """Helper to fetch observation data for an entire episode."""
        start = self.episode_starts[ep_idx]
        end = start + self.episode_lengths[ep_idx] + 1
        return self.observations[start:end]


    def get_obs(self, ep_idx: int, t: int) -> NDArray:
        """Helper to fetch observation data."""
        flat_idx = self.episode_starts[ep_idx] + t
        return self.observations[flat_idx]


    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
         """Fetches a data sample with the given index."""
