from typing import Mapping

from .archs import _internal_registry


def add(arch):
    """Register a new architecture."""
    return _internal_registry.add(arch)


def get(id: str):
    """Get architecture by ID."""
    return _internal_registry.get(id)


def load_from_file(path: str, strict: bool = True):
    """Detect and load architecture from state dict."""
    return _internal_registry.load_from_file(path, strict)


def load_from_state_dict(state_dict: Mapping[str, object]):
    """Detect and load architecture from state dict."""
    return _internal_registry.load_from_state_dict(state_dict)


__all__ = ['add', 'get', 'load_from_file', 'load_from_state_dict']
