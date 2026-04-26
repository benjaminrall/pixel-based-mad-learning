from __future__ import annotations

import torch
from torch import optim
from torch.types import Tensor
from tqdm import tqdm

from src.utils import MadDistConstraintDatasetView

from .objective import Objective
from src.models import Model

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from src import Trainer
    from src.datasets import MadDistDataset
    from src.configs.objectives import MadDistConfig


class MadDist(Objective):
    """Objective using the MadDist algorithm, as proposed by Steccanella et al. [2026]."""

    cfg: MadDistConfig

    identifier = 'maddist'
    aliases = ['MadDist', 'mad', 'MAD']


    def __init__(self, cfg: MadDistConfig, state: dict | None = None) -> None:
        super().__init__(cfg, state)

        self.constraint_dataset = MadDistConstraintDatasetView(cast('MadDistDataset', self.dataset))

        # Determines the state dimension from the first dataset sample
        sample_state = self.dataset[0][0]
        state_dim = sample_state.shape[-1]

        # Creates representation model to be trained
        self.phi = Model.get(
            identifier=cfg.model,
            input_dim=state_dim,
            hidden_dims=cfg.hidden_dims,
            latent_dim=cfg.latent_dim,
        ).to(self.device)

        # Initialises AdamW optimiser for training
        self.optimiser = optim.AdamW(self.phi.parameters(), lr=self.cfg.lr)

        self.updates_completed = 0

        # Loads state if it was provided
        if self.state is not None:
            self.phi.load_state_dict(self.state['phi'])
            self.optimiser.load_state_dict(self.state['optimiser'])
            self.updates_completed = self.state['updates_completed']
        

    def get_state_dict(self) -> dict:
        state_dict = super().get_state_dict()
        state_dict['phi'] = self.phi.state_dict()
        state_dict['optimiser'] = self.optimiser.state_dict()
        state_dict['updates_completed'] = self.updates_completed
        return state_dict
    

    def get_d(self, state_i: Tensor, state_j: Tensor) -> Tensor:
        """Returns the distance between two states, using MadDist's simple quasimetric."""
        # Encodes states using the representation model
        z_i = self.phi(state_i)
        z_j = self.phi(state_j)
        
        # Implementation of the ReLU-based simple quasimetric function 
        relus = torch.relu(z_i - z_j)
        maxes = torch.max(relus, dim=-1).values
        means = torch.mean(relus, dim=-1)
        return self.cfg.alpha * maxes + (1 - self.cfg.alpha) * means
    

    def update(self,
               state_i: Tensor,
               state_j: Tensor,
               diff: Tensor,
               state_r: Tensor,
               state_c_i: Tensor,
               state_c_j: Tensor,
               diff_c: Tensor) -> tuple[float, float]:
        """Performs a single training step for a given batch of data."""
        # Main objective
        d_o = self.get_d(state_i, state_j)
        l_o = torch.square((d_o / diff) - 1).mean()
        
        # Contrastive loss
        d_r = self.get_d(state_i, state_r)
        l_r = torch.square(torch.relu(1 - d_r / self.cfg.d_max)).mean()
        
        # Constraint loss
        d_c = self.get_d(state_c_i, state_c_j)
        l_c = torch.square(torch.relu(d_c - diff_c)).mean()

        # Final composite loss
        loss = l_o + self.cfg.w_r * l_r + self.cfg.w_c * l_c

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item(), d_r.mean().item()
    

    def train(self, trainer: Trainer) -> None:
        base_iterator = self.dataset.infinite_iterator()
        constraint_iterator = self.constraint_dataset.infinite_iterator()

        for _ in tqdm(range(1 + self.updates_completed, self.cfg.total_updates + 1), initial=self.updates_completed, total=self.cfg.total_updates):
            state_is, state_js, diffs, state_rs = next(base_iterator)
            state_c_is, state_c_js, diff_cs = next(constraint_iterator)

            state_is = state_is.to(self.device)
            state_js = state_js.to(self.device)
            diffs = diffs.flatten().to(self.device)
            state_rs = state_rs.to(self.device)

            state_c_is = state_c_is.to(self.device)
            state_c_js = state_c_js.to(self.device)
            diff_cs = diff_cs.flatten().to(self.device)

            loss, avg_dist = self.update(state_is, state_js, diffs, state_rs, state_c_is, state_c_js, diff_cs)

            self.updates_completed += 1

            trainer.log('loss', loss, self.updates_completed)
            trainer.log('avg_dist', avg_dist, self.updates_completed)
            trainer.update(self.updates_completed)
        
        trainer.save_checkpoint(self.updates_completed)
        

    def encode(self, state: Tensor, batch_size: int = 512) -> Tensor:
        results = []

        self.phi.eval()
        with torch.no_grad():
            for i in range(0, state.size(0), batch_size):
                state_batch = state[i : i + batch_size].to(self.device)
                z_batch = self.phi(state_batch)
                results.append(z_batch.cpu())

        self.phi.train()
        return torch.cat(results, dim=0)
    

    def distance(self, state: Tensor, goal: Tensor, batch_size: int = 512) -> Tensor:
        results = []
        
        is_batched_goal = goal.size(0) > 1

        self.phi.eval()
        with torch.no_grad():
            for i in range(0, state.size(0), batch_size):
                state_batch = state[i : i + batch_size].to(self.device)
                
                if is_batched_goal:
                    goal_batch = goal[i : i + batch_size].to(self.device)
                else:
                    goal_batch = goal.to(self.device)

                dist_batch = self.get_d(state_batch.float(), goal_batch.float())
                results.append(dist_batch.cpu())

        self.phi.train()
        return torch.cat(results, dim=0)
