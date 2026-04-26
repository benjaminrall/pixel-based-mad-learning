from __future__ import annotations

from dataclasses import dataclass, field

from .callback_config import CallbackConfig

from src.callbacks import MazeVisualiser


@dataclass
class MazeVisualiserConfig(CallbackConfig['MazeVisualiser']):

    wandb_label: str = 'maze_visualisation'
    update_sample_ratio: float = 0.1
    final_sample_ratio: float = 1

    use_spatial_sampling: bool = True
    spatial_resolution: float = 0.5

    goals: list[tuple[float, float]] = field(default_factory=list)

    pos_info_key: str = 'pos'
    centre_positions: bool = True
    flip_y: bool = True

    auto_alpha: bool = True
    alpha: float = 0.1  
    force_equal_aspect_ratio: bool = True
    show_title: bool = True
    show_labels: bool = True
    show_goal: bool = True
    show_colorbar: bool = True
    show_legend: bool = False

    identifier = 'maze_visualiser'
    aliases = ['maze_traj_visualiser']

    def __post_init__(self) -> None:
        for i, goal in enumerate(self.goals):
            if isinstance(goal, list):
                if len(goal) != 2:
                    raise ValueError('Goals for the maze visualiser callback must be 2 dimensional.')
                self.goals[i] = (goal[0], goal[1])

    @property
    def target_class(self) -> type[MazeVisualiser]:
        return MazeVisualiser    
