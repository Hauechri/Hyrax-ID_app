from __future__ import annotations
from .base import Representation

_REGISTRY: dict[str, type[Representation]] = {}


def register(cls: type[Representation]) -> type[Representation]:
    if cls.name in _REGISTRY:
        raise ValueError(f"Duplicate representation name: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type[Representation]:
    return _REGISTRY[name]


def create(name: str, **params) -> Representation:
    return _REGISTRY[name](**params)


def available() -> dict[str, type[Representation]]:
    return dict(_REGISTRY)
