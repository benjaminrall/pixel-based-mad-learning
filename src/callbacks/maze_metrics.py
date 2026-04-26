from __future__ import annotations

import torch
import wandb
import numpy as np
import networkx as nx
from numpy.typing import NDArray
from scipy.stats import spearmanr, pearsonr

from .callback import Callback

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src import Trainer
    from src.objectives import Objective
    from src.configs.callbacks import MazeMetricsConfig


class MazeMetrics(Callback):
    """Callback to calculate quantitative metrics for maze environments."""

    identifier = 'maze_metrics'

    cfg: MazeMetricsConfig


    def __init__(self, cfg: MazeMetricsConfig) -> None:
        super().__init__(cfg)
        self._initialised = False


    def _build_maze_graph(self):
        """Constructs a networkx graph from the maze layout and precomputes a matrix of all ground-truth distances."""
        num_rows = len(self.cfg.maze_layout)
        num_cols = len(self.cfg.maze_layout[0])

        graph = nx.grid_2d_graph(num_rows, num_cols)

        for r in range(num_rows):
            for c in range(num_cols):
                if self.cfg.maze_layout[r][c] == 1:
                    graph.remove_node((r, c))

        self.distances = np.full((num_rows, num_cols, num_rows, num_cols), -1, dtype=np.float32)

        for (r1, c1), targets in nx.all_pairs_shortest_path_length(graph):
            for (r2, c2), dist in targets.items():
                self.distances[r1, c1, r2, c2] = dist * self.cfg.maze_scale

        self._initialised = True


    def _pos_to_grid(self, positions: NDArray) -> tuple[NDArray, NDArray]:
        """Converts an array of real (x, y) positions to row and column grid index arrays."""
        x = positions[:, 0]
        y = positions[:, 1]

        ref_x, ref_y = self.cfg.global_ref_point
        ref_row, ref_col = self.cfg.maze_ref_point

        cols = ref_col + np.round((x - ref_x) / self.cfg.maze_scale).astype(np.int32)
        rows = ref_row + np.round((y - ref_y) / self.cfg.maze_scale).astype(np.int32)

        return rows, cols


    def on_train_start(self, trainer: Trainer, objective: Objective) -> None:
        self._build_maze_graph()

        if not trainer.cfg.track_wandb or wandb.run is None:
            return
        
        metrics = self.compute_metrics(objective)
        for key in metrics:
            trainer.log(f'{self.cfg.wandb_label}/{key}', metrics[key], 0)


    def on_train_end(self, trainer: Trainer, objective: Objective) -> None:
        if trainer.objective.cfg.total_updates % self.cfg.callback_interval == 0:
            return
        
        if not trainer.cfg.track_wandb or wandb.run is None:
            return
        
        metrics = self.compute_metrics(objective)
        for key in metrics:
            trainer.log(f'{self.cfg.wandb_label}/{key}', metrics[key], trainer.objective.cfg.total_updates)


    def on_update(self, trainer: Trainer, objective: Objective, progress: int) -> None:
        if progress % self.cfg.callback_interval != 0:
            return
        
        if not trainer.cfg.track_wandb or wandb.run is None:
            return
        
        metrics = self.compute_metrics(objective)
        for key in metrics:
            trainer.log(f'{self.cfg.wandb_label}/{key}', metrics[key], progress)


    def compute_metrics(self, objective: Objective):
        """Computes and returns the quantitative metrics for the given objective."""

        if not self._initialised:
            self._build_maze_graph()

        dataset = objective.dataset
        total_obs = len(dataset)

        # Samples random initial and goal states
        indices_s = np.random.randint(0, total_obs, size=self.cfg.num_samples)
        indices_g = np.random.randint(0, total_obs, size=self.cfg.num_samples)

        # Gets the real positions of the states and converts them to grid rows and columns
        positions_s = dataset.infos['pos'][indices_s]
        positions_g = dataset.infos['pos'][indices_g]
        rows_s, cols_s = self._pos_to_grid(positions_s)
        rows_g, cols_g = self._pos_to_grid(positions_g)

        # Gets valid ground-truth distances for the sampled states
        all_true_dists = self.distances[rows_s, cols_s, rows_g, cols_g]
        valid_mask = all_true_dists > 0
        true_dists = all_true_dists[valid_mask]
        valid_s_idx = indices_s[valid_mask]
        valid_g_idx = indices_g[valid_mask]

        # Gets the actual state observations from the dataset
        obs_s_np = dataset.observations[valid_s_idx]
        obs_g_np = dataset.observations[valid_g_idx]
        obs_s = torch.from_numpy(obs_s_np).to(dtype=torch.float32)
        obs_g = torch.from_numpy(obs_g_np).to(dtype=torch.float32)

        # Computes the predicted distances corresponding to the true distances
        pred_dists = objective.distance(obs_s, obs_g).numpy()

        # Spearman Correlation 
        spearman, _ = spearmanr(true_dists, pred_dists)

        # Pearson Correlation 
        pearson, _ = pearsonr(true_dists, pred_dists)

        # Ratio Coefficient of Variation
        ratios = pred_dists / true_dists
        ratio_cv = np.std(ratios) / np.mean(ratios)

        return {
            'spearman': spearman,
            'pearson': pearson,
            'ratio_cv': ratio_cv,
        }
