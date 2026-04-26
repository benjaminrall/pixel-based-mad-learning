from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.types import Tensor

from src.utils import Registrable
from src.utils import get_device

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src import Trainer
    from src.configs.objectives import ObjectiveConfig


class Objective(nn.Module, Registrable['Objective'], ABC):
    """Abstract base class for all trainable objectives."""

    def __init__(self, cfg: ObjectiveConfig, state: dict | None = None):
        super().__init__()
        self.cfg = cfg
        self.state = state
        self.device = get_device()
        
        # Builds the dataset to train on
        self.dataset = self.cfg.dataset.build()


    def save(self, path: str) -> None:
        """Saves the objective's dictionary to the specified path."""
        torch.save(self.to_dict(), path)


    def to_dict(self) -> dict:
        """Serialises an objective to a dictionary."""
        return {
            'type': type(self),
            'params': {
                'cfg': self.cfg,
                'state': self.get_state_dict()
            }
        }
    

    @staticmethod
    def from_dict(d: dict) -> Objective:
        """Deserialises an objective from a dictionary."""
        return d['type'](**d['params'])


    @staticmethod
    def load_checkpoint(filepath: str) -> Objective:
        """Loads an objective from a Trainer checkpoint with the specified path."""
        # Loads the checkpoint dictionary from the path
        checkpoint = torch.load(filepath, weights_only=False)

        # Gets the trainable objective from the objective dictionary
        return Objective.from_dict(checkpoint['objective'])


    @abstractmethod
    def get_state_dict(self) -> dict:
        """Returns the state dict required to restore an objective."""
        return {}


    @abstractmethod
    def train(self, trainer: Trainer):
        """
        Trains the objective, using the given Trainer 
        instance for logging and checkpoints.
        """


    @abstractmethod
    def encode(self, state: Tensor, batch_size: int = 512) -> Tensor:
        """
        Encodes states into a latent representation.
        Processes data in batches and runs in inference mode.
        """


    @abstractmethod
    def distance(self, state: Tensor, goal: Tensor, batch_size: int = 512) -> Tensor:
        """
        Calculates the distance between states and goals.
        Processes data in batches and runs in inference mode.
        """
