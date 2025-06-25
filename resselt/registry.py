import os
import pickle  # noqa: S403
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import TypeVar

import torch
from typing_extensions import override

from .factory import Architecture
from .utilities.state_dict import canonicalize_state_dict


class ArchitectureNotFound(Exception):
    pass


T = TypeVar('T', bound=torch.nn.Module, covariant=True)  # noqa: PLC0105

safe_list = {
    ('collections', 'OrderedDict'),
    ('typing', 'OrderedDict'),
    ('torch._utils', '_rebuild_tensor_v2'),
    ('torch', 'BFloat16Storage'),
    ('torch', 'FloatStorage'),
    ('torch', 'HalfStorage'),
    ('torch', 'IntStorage'),
    ('torch', 'LongStorage'),
    ('torch', 'DoubleStorage'),
}


class RestrictedUnpickler(pickle.Unpickler):  # noqa: S301
    @override
    def find_class(self, module: str, name: str):
        # Only allow required classes to load state dict
        if (module, name) not in safe_list:
            msg = f"Global '{module}.{name}' is forbidden"
            raise pickle.UnpicklingError(msg)
        return super().find_class(module, name)


RestrictedUnpickle = SimpleNamespace(
    Unpickler=RestrictedUnpickler, __name__='pickle', load=lambda *args, **kwargs: RestrictedUnpickler(*args, **kwargs).load()
)


class Registry:
    def __init__(self):
        self.store: dict[str, Architecture] = {}

    def __contains__(self, uid: str):
        return uid in self.store

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

    def get(self, uid: str) -> Architecture:
        architecture = self.store[uid]
        if not architecture:
            raise ArchitectureNotFound
        return architecture

    def load_from_file(self, path: str) -> T:
        extension = os.path.splitext(path)[1].lower()
        if extension == '.pt':
            try:
                state_dict = torch.jit.load(path).state_dict()
            except RuntimeError:
                try:
                    pth_state_dict = torch.load(path, pickle_module=RestrictedUnpickle)
                except Exception:  # noqa: BLE001
                    pth_state_dict = None

                if pth_state_dict is None:
                    raise

                state_dict = pth_state_dict

        elif extension == '.pth' or extension == '.ckpt':
            state_dict = torch.load(path, pickle_module=RestrictedUnpickle)
        elif extension == '.safetensors':
            import safetensors.torch  # noqa: PLC0415

            state_dict = safetensors.torch.load_file(path)
        else:
            msg = f'Unsupported model file extension {extension}. Please try a supported model type.'
            raise ValueError(msg)

        return self.load_from_state_dict(state_dict)

    def load_from_state_dict(self, state_dict: Mapping[str, object]) -> T:
        state_dict = canonicalize_state_dict(state_dict)

        for architecture in self.store.values():
            is_valid = architecture.detect(state_dict)
            if is_valid:
                model = architecture.load(state_dict)
                model.load_state_dict(state_dict)
                return model

        raise ArchitectureNotFound
