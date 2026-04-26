from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic, Any

from src.utils import Registrable


class Configurable(Protocol):
    """Protocol defining the constructor signature for classes built from a config."""
    def __init__(self, cfg: Any) -> None: ...

# Generic variables for configs and configurable classes
T = TypeVar('T', bound='Configurable', covariant=True)
C = TypeVar('C', bound='Config')


@dataclass
class Config(Generic[T], ABC):
    """Abstract base class for all config classes."""

    @property
    @abstractmethod
    def target_class(self) -> type[T]:
        """Returns the target class type for the config."""


    @classmethod
    def from_dict(cls: type[C], d: dict, id_key: str = 'type') -> C:
        """
        Returns an instance of the config constructed from a dictionary. 
        
        If the config has a registry, the relevant subclass to construct
        is determined using the provided identifier key.
        """
        # Handles config base classes which have a registry
        if issubclass(cls, Registrable) and Registrable in cls.__bases__:
            if id_key not in d:
                raise ValueError(
                    f"Construction dictionary for {cls.__name__} is missing "
                    f"required identifier key: '{id_key}'"
                )
            d = d.copy()
            identifier = d.pop(id_key)
            return cls.get(identifier, **d)

        # Constructs generic config classes by unpacking the dictionary        
        return cls(**d)


    def build(self, **kwargs) -> T:
        """Instantiates an instance of the target class, constructed using its config."""
        return self.target_class(cfg=self, **kwargs)
