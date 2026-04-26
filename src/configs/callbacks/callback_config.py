from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TypeVar

from src.configs import Config
from src.utils import Registrable
from src.callbacks import Callback

C = TypeVar('C', bound='Callback', covariant=True)


@dataclass
class CallbackConfig(Config[C], Registrable['CallbackConfig'], ABC):
    """Abstract base class for config classes for all trainer callbacks."""

    callback_interval: int = 1
