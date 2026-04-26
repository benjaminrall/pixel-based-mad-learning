from __future__ import annotations

from dataclasses import dataclass, field

from .callback_config import CallbackConfig

from src.callbacks import MazeMetrics


@dataclass
class MazeMetricsConfig(CallbackConfig['MazeMetrics']):

    wandb_label: str = 'maze_metrics'

    pos_info_key: str = 'pos'
    num_samples: int = 1024

    maze_layout: list[list[int]] = field(default_factory=list)
    maze_scale: float = 1
    global_ref_point: tuple[float, float] = (0, 0)
    maze_ref_point: tuple[int, int] = (1, 1)

    identifier = 'maze_metrics'

    @property
    def target_class(self) -> type[MazeMetrics]:
        return MazeMetrics
