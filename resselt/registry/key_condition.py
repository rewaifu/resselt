from __future__ import annotations

from typing import Literal, Mapping


class KeyCondition:
    """
    A condition that checks if a state dict has the given keys.
    """

    def __init__(self, kind: Literal['all', 'any'], keys: tuple[str | KeyCondition, ...]):
        self._keys = keys
        self._kind: Literal['all', 'any'] = kind

    @staticmethod
    def has_all(*keys: str | KeyCondition) -> KeyCondition:
        return KeyCondition('all', keys)

    @staticmethod
    def has_any(*keys: str | KeyCondition) -> KeyCondition:
        return KeyCondition('any', keys)

    def __call__(self, state_dict: Mapping[str, object]) -> bool:
        def _detect(key: str | KeyCondition) -> bool:
            if isinstance(key, KeyCondition):
                return key(state_dict)
            return key in state_dict

        if self._kind == 'all':
            return all(_detect(key) for key in self._keys)
        else:
            return any(_detect(key) for key in self._keys)
