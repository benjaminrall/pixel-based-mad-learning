from __future__ import annotations

import time
import random
import os
import torch
import numpy as np
import wandb
import yaml
from torch.utils.tensorboard.writer import SummaryWriter

from src.objectives import Objective
from src.utils import to_nested_dict, get_random_state, restore_random_state

from src.configs import TrainerConfig

from dotenv import load_dotenv
load_dotenv()


class Trainer:
    """Class that wraps trainable objectives to provide logging and checkpointing."""

    def __init__(self, cfg: TrainerConfig, saved_objective: Objective | None = None) -> None:
        self.cfg = cfg
        self.trained = False
        self.objective = saved_objective or self.get_objective()
        self.callbacks = [c.build() for c in self.cfg.callbacks]

        if cfg.run_id == "auto":
            self.run_name = f"{self.cfg.run_name}-{int(time.time())}"
        elif cfg.run_id == "":
            self.run_name = self.cfg.run_name
        else:
            self.run_name = f"{self.cfg.run_name}-{self.cfg.run_id}"

        self._reset_seed()


    def _reset_seed(self) -> None:
        """Resets all relevant random states using the trainer's seed for deterministic training."""
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True


    def _init_wandb(self) -> None:
        """Attempts to sign-in to W&B and initialise a new run."""

        if 'WANDB_API_KEY' not in os.environ:
            raise RuntimeError(
                "The 'WANDB_API_KEY' environment variable must "
                "be set to use W&B tracking."
            )
        
        wandb.login(key=os.environ['WANDB_API_KEY'])
        self.run = wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            sync_tensorboard=True,
            name=self.run_name,
            config=to_nested_dict(self.cfg),
            monitor_gym=True,
            dir=self.cfg.wandb_dir
        )


    @classmethod
    def from_yaml(cls, filepath: str) -> Trainer:
        """Constructs a Trainer object from a given YAML config file path."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            cfg = TrainerConfig.from_dict(data)
        return cls(cfg)


    def save_checkpoint(self, progress: int) -> None:
        """Saves a checkpoint of the current training progress."""

        if not self.trained:
            raise RuntimeError("Checkpoint cannot be saved for an unused Trainer instance.")
        
        # Ensures the checkpoint folder exists
        checkpoint_folder = os.path.join(self.cfg.checkpoint_folder, self.run_name)
        os.makedirs(checkpoint_folder, exist_ok=True)

        # Creates the checkpoint dictionary
        checkpoint = {
            'cfg': self.cfg,
            'objective': self.objective.to_dict(),
            'random_state': get_random_state()
        }

        # Saves the checkpoint using torch
        torch.save(checkpoint, os.path.join(checkpoint_folder, f"{progress}.pt"))


    @classmethod
    def load_checkpoint(cls, filepath: str) -> Trainer:
        """Loads a checkpoint of training progress from the specified path."""
        # Loads the checkpoint dictionary from the path
        checkpoint = torch.load(filepath, weights_only=False)

        # Gets trainable objective from a dictionary
        objective = Objective.from_dict(checkpoint['objective'])

        # Creates the trainer instance from the saved config and objective
        trainer = cls(checkpoint['cfg'], saved_objective=objective)

        # Restores random state to continue training
        restore_random_state(checkpoint['random_state'])
        return trainer


    def get_objective(self) -> Objective:
        """Builds and returns the trainer's objective."""
        return self.cfg.objective.build()


    def log(self, tag: str, value, global_step: int | None = None) -> None:
        """Logs the given value to Tensorboard as a scalar."""
        if not self.trained:
            raise RuntimeError('Cannot log to an unused Trainer instance.')
        self.writer.add_scalar(tag, value, global_step)


    def update(self, progress: int) -> None:
        """Updates the trainer with current training progress, used for checkpointing and callbacks."""
        if not self.trained:
            raise RuntimeError("Updates cannot be given to an unused Trainer instance.")
        
        for callback in self.callbacks:
            callback.on_update(self, self.objective, progress)

        if self.cfg.save_checkpoints and progress % self.cfg.checkpoint_interval == 0:
            self.save_checkpoint(progress)


    def train(self) -> None:
        """Starts training the Trainer's objective."""
        # Prevents a trainer instance from being used more than once
        if self.trained:
            raise RuntimeError("Trainer instance has already been used - reuse is not allowed.")
        self.trained = True

        # Initialises W&B run and Tensorboard writer for the training run
        if self.cfg.track_wandb:
            self._init_wandb()
        self.writer = SummaryWriter(f'logs/tensorboard/{self.run_name}')

        self._reset_seed()

        for callback in self.callbacks:
            callback.on_train_start(self, self.objective)

        self.objective.train(self)
 
        for callback in self.callbacks:
            callback.on_train_end(self, self.objective)

        if self.cfg.track_wandb:
            wandb.finish()
