from __future__ import annotations

import torch
import wandb
import numpy as np
from numpy.typing import NDArray

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .callback import Callback

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src import Trainer
    from src.objectives import Objective
    from src.configs.callbacks import MazeVisualiserConfig


class MazeVisualiser(Callback):
    """Callback to visualise learned representations in maze environments."""

    identifier = 'maze_visualiser'
    aliases = ['maze_traj_visualiser']

    cfg: MazeVisualiserConfig


    def __init__(self, cfg: MazeVisualiserConfig) -> None:
        super().__init__(cfg)
        self._initialised = False


    def _spatially_uniform_sample(self, positions: NDArray, target_size: int) -> NDArray:
        """Provides a spatially uniform subsample of the given positions array."""
        
        total_points = len(positions)

        shuffled_indices = np.random.permutation(total_points)
        shuffled_pos = positions[shuffled_indices]

        quantised = np.round(shuffled_pos / self.cfg.spatial_resolution).astype(np.int32)

        min_x = quantised[:, 0].min()
        max_x_range = quantised[:, 0].max() - min_x + 1
        hashed_coords = quantised[:, 1] * max_x_range + (quantised[:, 0] - min_x)

        _, unique_first_indices = np.unique(hashed_coords, return_index=True)

        spatially_sampled_indices = shuffled_indices[unique_first_indices]

        if len(spatially_sampled_indices) >= target_size:
            # More unique covered cells than needed; downsample randomly
            return np.random.choice(spatially_sampled_indices, size=target_size, replace=False)
        else:
            # Not enough unique cells to fill the ratio, pad with standard random samples
            shortfall = target_size - len(spatially_sampled_indices)
            
            # Mask out the indices we've already picked to avoid duplicates
            mask = np.ones(total_points, dtype=bool)
            mask[spatially_sampled_indices] = False
            remaining_pool = np.arange(total_points)[mask]
            
            padding_indices = np.random.choice(remaining_pool, size=shortfall, replace=False)
            return np.concatenate([spatially_sampled_indices, padding_indices])


    def _init_visualiser(self, objective: Objective, sample_ratio: float = 1):
        """Initialises the visualiser for a given objective and sample ratio."""
        dataset = objective.dataset
        self.positions = np.copy(dataset.infos[self.cfg.pos_info_key])

        # Centres positions
        if self.cfg.centre_positions:
            center = (self.positions.max(axis=0) + self.positions.min(axis=0)) / 2.0
            self.positions -= center

        # Flips position y coords
        if self.cfg.flip_y:
            self.positions[:, 1] = -self.positions[:, 1]

        # Computes reference goal observations and positions
        self.ref_goal_obs = []
        for goal in self.cfg.goals:
            target_coord = np.array(goal)

            distances = np.linalg.norm(self.positions - target_coord, axis=1)
            ref_idx = np.argmin(distances)

            ref_obs = torch.from_numpy(dataset.observations[ref_idx])
            ref_pos = self.positions[ref_idx]

            self.ref_goal_obs.append((ref_obs.unsqueeze(0), ref_pos))

        # Stores sampled observations and positions using the given sample ratio
        total_obs = len(dataset)
        sample_size = int(total_obs * sample_ratio)

        if self.cfg.use_spatial_sampling:
            indices = self._spatially_uniform_sample(self.positions, sample_size)
        else:
            indices = np.random.choice(total_obs, size=sample_size, replace=False)

        self.current_pos = self.positions[indices] if sample_ratio < 1 else self.positions
        obs_np = dataset.observations[indices] if sample_ratio < 1 else dataset.observations
        self.current_obs = torch.from_numpy(obs_np)

        # Stores auto alpha for the number of positions to be plotted
        # Tuned for 1000000 points -> alpha of 0.1
        num_points = len(self.current_pos)
        if self.cfg.auto_alpha and num_points > 0:
            calculated_alpha = (self.cfg.alpha * 1000.0) / np.sqrt(num_points)
            self.alpha = max(0.001, min(1.0, calculated_alpha))
        else:
            self.alpha = self.cfg.alpha

        self._initialised = True


    def on_train_start(self, trainer: Trainer, objective: Objective) -> None:
        self._init_visualiser(objective, self.cfg.update_sample_ratio)

        if not trainer.cfg.track_wandb or wandb.run is None:
            return
        
        images = []
        for i in range(len(self.cfg.goals)):
            fig = self.render(objective, i)
            images.append(wandb.Image(fig, caption=f'Goal {i+1} at start of training'))
            plt.close(fig)
        
        wandb.log({self.cfg.wandb_label: images})


    def on_train_end(self, trainer: Trainer, objective: Objective) -> None:
        if not trainer.cfg.track_wandb or wandb.run is None:
            return
        
        self._init_visualiser(objective, self.cfg.final_sample_ratio)

        images = []
        for i in range(len(self.cfg.goals)):
            fig = self.render(objective, i)
            images.append(wandb.Image(fig, caption=f'Goal {i+1} at end of training'))
            plt.close(fig)
        
        wandb.log({self.cfg.wandb_label: images})


    def on_update(self, trainer: Trainer, objective: Objective, progress: int) -> None:
        if progress % self.cfg.callback_interval != 0:
            return
        
        if not trainer.cfg.track_wandb or wandb.run is None:
            return
        
        images = []
        for i in range(len(self.cfg.goals)):
            fig = self.render(objective, i)
            images.append(wandb.Image(fig, caption=f'Goal {i+1} at step {progress}'))
            plt.close(fig)
        
        wandb.log({self.cfg.wandb_label: images})


    def render(self, objective: Objective, goal_idx: int) -> Figure:
        """Renders a visualisation of the given objective for the given goal index."""
        if not self._initialised:
            raise Exception('Cannot render using uninitialised maze trajectory visualiser.')
        
        goal_obs, goal_pos = self.ref_goal_obs[goal_idx]

        distances = objective.distance(self.current_obs, goal_obs).numpy()

        fig, ax = plt.subplots(figsize=(6, 5))
        if self.cfg.force_equal_aspect_ratio:
            ax.set_aspect('equal', adjustable='box')
    
        if self.cfg.show_title:
            ax.set_title(f'Distance Map for Goal {goal_idx + 1}')
        
        if self.cfg.show_labels:
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        scatter = ax.scatter(self.current_pos[:, 0], self.current_pos[:, 1], c=distances, cmap='magma_r', s=2, alpha=self.alpha)

        if self.cfg.show_colorbar:
            cbar = fig.colorbar(scatter, label='Distance')
            if cbar.solids:
                cbar.solids.set_alpha(1)

        if self.cfg.show_goal:
            ax.scatter(goal_pos[0], 
                    goal_pos[1],
                    color='red', marker='*', s=200, label='Goal',
                    edgecolor='white', linewidth=1)

        if self.cfg.show_legend:
            ax.legend()

        fig.tight_layout()
        return fig
