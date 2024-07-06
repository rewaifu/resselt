from typing import Mapping, Dict, Iterator

from resselt.registry.architecture import Architecture, WrappedModel
from resselt.utils.state_dict import canonicalize_state_dict


class ArchitectureNotFound(Exception):
    pass


class Registry:
    def __init__(self):
        self.store: Dict[str, Architecture] = {}

    def __contains__(self, id: str):
        return id in self.store

    def __iter__(self) -> Iterator[Architecture]:
        self._iter_keys = iter(self.store)
        return self

    def __next__(self) -> Architecture:
        if self._iter_keys is None:
            raise StopIteration
        try:
            key = next(self._iter_keys)
            return self.store[key]
        except StopIteration:
            self._iter_keys = None
            raise

    def add(self, cls: Architecture):
        self.store[cls.id] = cls

    def get(self, id: str) -> Architecture:
        architecture = self.store[id]
        if not architecture:
            raise ArchitectureNotFound
        return architecture

    def load_from_state_dict(self, state_dict: Mapping[str, object]) -> WrappedModel:
        state_dict = canonicalize_state_dict(state_dict)

        for architecture in self.store.values():
            is_valid = architecture.detect(state_dict)
            if is_valid:
                wrapped_model = architecture.load(state_dict)
                wrapped_model.load_state_dict(state_dict)
                return wrapped_model

        raise ArchitectureNotFound


__all__ = ['Registry', 'ArchitectureNotFound']
