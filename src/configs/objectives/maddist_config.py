from __future__ import annotations

from dataclasses import dataclass, field

from .objective_config import ObjectiveConfig

from src.objectives import MadDist


@dataclass
class MadDistConfig(ObjectiveConfig['MadDist']):

    model: str = 'maddist'
    
    hidden_dims: list[int] = field(default_factory=lambda:[512, 512, 256])
    latent_dim: int = 512
    lr: float = 1e-4
    alpha: float = 0.5
    w_r: float = 1
    w_c: float = 0.1
    d_max: float = 100

    identifier = 'maddist'
    aliases = ['MadDist', 'mad', 'MAD']

    @property
    def target_class(self) -> type[MadDist]:
        return MadDist
