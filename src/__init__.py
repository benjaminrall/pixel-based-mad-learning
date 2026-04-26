"""Codebase for my final-year research project, entitled 'Learning the Minimum Action Distance in Pixel-Based Environments'."""

from .trainer import Trainer

from . import utils
from . import datasets
from . import objectives
from . import configs
from . import callbacks
from . import models

__all__ = [
    'callbacks',
    'configs',
    'datasets',
    'models',
    'objectives',
    'utils',
    'Trainer',
]
