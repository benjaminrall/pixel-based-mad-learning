from __future__ import annotations

from dataclasses import dataclass, field

from .objective_config import ObjectiveConfig

from src.objectives import HILPEncoder


@dataclass
class HILPConfig(ObjectiveConfig['HILPEncoder']):

    model: str = 'hilp'
    
    hidden_dims: list[int] = field(default_factory=lambda:[512, 512])
    latent_dim: int = 32
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.9
    polyak: float = 0.005

    identifier = 'hilp'
    aliases = ['hilp_encoder', 'HILPEncoder', 'HILP']

    @property
    def target_class(self) -> type[HILPEncoder]:
        return HILPEncoder
