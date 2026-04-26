from abc import ABC, abstractmethod
from src.utils import Registrable
from torch import nn
from torch.types import Tensor


class Model(nn.Module, Registrable['Model'], ABC):
    """Base class for all torch models."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.model = self._construct_model(**kwargs)


    @abstractmethod
    def _construct_model(self, **kwargs) -> nn.Module:
        """Constructs and returns the model."""


    def forward(self, x: Tensor) -> Tensor:
        """Passes an input forward through the model."""
        return self.model(x)
