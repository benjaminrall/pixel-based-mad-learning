from __future__ import annotations
from typing import TypeVar, Generic, cast
import inspect

T = TypeVar('T', covariant=True)


class Registrable(Generic[T]):
    """A mixin class that provides a class with a registry of its subclasses."""

    _registry: dict[str, type[T]] = {}
    """Dictionary mapping identifiers to their corresponding subclass type."""

    identifier: str
    """The string identifier of the subclass."""

    aliases: list[str] = []
    """Optional list of additional string identifiers of the subclass."""


    def __init_subclass__(cls) -> None:
        """Initialisees a separate registry for each base class family"""
        super().__init_subclass__()

        # Direct children of Registrable get their own registry
        if Registrable in cls.__bases__:
            cls._registry = {}

        # Skips registration for abstract classes
        if inspect.isabstract(cls):
            return
        
        # Ensures the class has an explicitly defined identifier
        if not hasattr(cls, 'identifier') or not isinstance(cls.identifier, str):
            raise AttributeError(
                f"Class {cls.__name__} must define a unique 'identifier' string "
                f"to be properly registered."
            )

        for identifier in [cls.identifier] + cls.aliases:
            # Ensures class identifiers and aliases are unique
            if identifier in cls._registry:
                existing = cls._registry[identifier].__name__
                raise TypeError(
                    f"Cannot register class '{cls.__name__}': its identifier "
                    f"'{identifier}' is already used by class '{existing}'."
                )
            cls._registry[identifier] = cast(type[T], cls)


    @classmethod
    def get_registry(cls) -> dict[str, type[T]]:
        """Returns the registry specific to this class family."""
        return cls._registry
    

    @classmethod
    def get_type(cls, identifier: str) -> type[T]:
        """Retrieves a class type from a string identifier."""        
        registry = cls.get_registry()
        if identifier in registry:
            return registry[identifier]
        
        raise ValueError(
            f"Unknown {cls.__name__}: '{identifier}'.\n\t"
            f"Available options are: {', '.join(list(registry.keys()))}"
        )
    

    @classmethod
    def get(cls, identifier: str | T, **kwargs) -> T:
        """
        Returns an instance of a class from a string identifier, 
        constructed with the given kwargs.
        """        
        if not isinstance(identifier, str):
            return identifier
        return cls.get_type(identifier)(**kwargs)
