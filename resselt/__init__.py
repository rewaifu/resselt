from typing import Mapping

from .archs import internal_registry


def add(arch):
    """Register a new architecture."""
    return internal_registry.add(arch)


def get(id: str):
    """Get architecture by ID."""
    return internal_registry.get(id)


def load_from_file(path: str):
    """Detect and load architecture from state dict."""
    return internal_registry.load_from_file(path)


def load_from_state_dict(state_dict: Mapping[str, object]):
    """Detect and load architecture from state dict."""
    return internal_registry.load_from_state_dict(state_dict)


__all__ = ['add', 'get', 'load_from_file', 'load_from_state_dict']
