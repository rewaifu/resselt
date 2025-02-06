from typing import Mapping, Dict, Iterator, TypeVar

import torch

from .factory import Architecture
from .utilities.state_dict import canonicalize_state_dict


class ArchNotFound(Exception):
    pass


T = TypeVar('T', bound=torch.nn.Module, covariant=True)


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
            raise ArchNotFound
        return architecture

    def load_from_file(self, path: str, strict: bool = True) -> T:
        if strict:
            from torch.serialization import add_safe_globals
            from typing import OrderedDict

            add_safe_globals([OrderedDict])
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, weights_only=False)
        return self.load_from_state_dict(state_dict)

    def load_from_state_dict(self, state_dict: Mapping[str, object]) -> T:
        state_dict = canonicalize_state_dict(state_dict)

        for architecture in self.store.values():
            is_valid = architecture.detect(state_dict)
            if is_valid:
                model = architecture.load(state_dict)
                model.load_state_dict(state_dict)
                return model

        raise ArchNotFound
