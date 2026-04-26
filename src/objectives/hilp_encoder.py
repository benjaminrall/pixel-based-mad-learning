from __future__ import annotations

import copy
import torch
from torch import optim
from torch.types import Tensor
from tqdm import tqdm

from .objective import Objective
from src.models import Model

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src import Trainer
    from src.configs.objectives import HILPConfig


class HILPEncoder(Objective):
    """Objective using the HILP representation learning objective, as proposed by Park et al. [2024]."""

    cfg: HILPConfig

    identifier = 'hilp'
    aliases = ['hilp_encoder', 'HILPEncoder', 'HILP']


    def __init__(self, cfg: HILPConfig, state: dict | None = None):
        super().__init__(cfg, state)

        # Determines the state dimension from the first dataset sample
        sample_state, _, _ = self.dataset[0]
        state_dim = sample_state.shape[-1]

        # Creates HILP representation model to be trained
        self.phi = Model.get(
            identifier=cfg.model,
            input_dim=state_dim,
            hidden_dims=cfg.hidden_dims,
            latent_dim=cfg.latent_dim,
        ).to(self.device)

        # Initialises target network 
        self.phi_target = copy.deepcopy(self.phi)
        for p in self.phi_target.parameters():
            p.requires_grad = False

        # Initialises Adam optimiser for HILP training
        self.optimiser = optim.Adam(self.phi.parameters(), lr=self.cfg.lr)

        self.updates_completed = 0

        # Loads state if it was provided
        if self.state is not None:
            self.phi.load_state_dict(self.state['phi'])
            self.phi_target.load_state_dict(self.state['phi_target'])
            self.optimiser.load_state_dict(self.state['optimiser'])
            self.updates_completed = self.state['updates_completed']
        

    def get_state_dict(self) -> dict:
        state_dict = super().get_state_dict()
        state_dict['phi'] = self.phi.state_dict()
        state_dict['phi_target'] = self.phi_target.state_dict()
        state_dict['optimiser'] = self.optimiser.state_dict()
        state_dict['updates_completed'] = self.updates_completed
        return state_dict
    

    def get_v(self, state: Tensor, goal: Tensor, use_target: bool = False) -> Tensor:
        """Calculates value V(s, g) = -||phi(s) - phi(g)||"""
        net = self.phi_target if use_target else self.phi
        z_s = net(state)
        z_g = net(goal)
        v = -torch.norm(z_s - z_g + 1e-6, dim=-1)
        return v
    

    def expectile_loss(self, u: Tensor) -> Tensor:
        """Computes the asymmetric expectile loss function."""
        weight = torch.where(u > 0, self.cfg.tau, 1 - self.cfg.tau)
        return (weight * (u ** 2)).mean()


    def update(self, state: Tensor, next_state: Tensor, goal: Tensor) -> tuple[float, float]:
        """Performs a single training step for a given batch of data."""
        with torch.no_grad():
            target_v = self.get_v(next_state, goal, use_target=True)

        curr_v = self.get_v(state, goal, use_target=False)

        u = -1.0 + self.cfg.gamma * target_v - curr_v
        loss = self.expectile_loss(u)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self._polyak_update()

        return loss.item(), -curr_v.mean().item()


    def _polyak_update(self):
        """Applies a soft update to the target network."""
        for param, target_param in zip(self.phi.parameters(), self.phi_target.parameters()):
            target_param.data.mul_(1 - self.cfg.polyak)
            torch.add(target_param.data, param.data, alpha=self.cfg.polyak, out=target_param.data)


    def train(self, trainer: Trainer):
        iterator = self.dataset.infinite_iterator()
        for _ in tqdm(range(1 + self.updates_completed, self.cfg.total_updates + 1), initial=self.updates_completed, total=self.cfg.total_updates):
            states, next_states, goals = next(iterator)

            states = states.to(self.device)
            next_states = next_states.to(self.device)
            goals = goals.to(self.device)

            loss, avg_dist = self.update(states, next_states, goals)

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

                dist_batch = -self.get_v(state_batch.float(), goal_batch.float())
                results.append(dist_batch.cpu())

        self.phi.train()
        return torch.cat(results, dim=0)
