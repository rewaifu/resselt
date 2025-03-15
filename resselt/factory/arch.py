from dataclasses import dataclass

import torch
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Mapping

from .key_condition import KeyCondition

T = TypeVar('T', bound=torch.nn.Module, covariant=True)


@dataclass
class ModelMetadata:
    """Mixin for adding SR model specific attributes and methods"""

    in_channels: int
    out_channels: int
    upscale: int
    name: str


class Architecture(ABC, Generic[T]):
    def __init__(self, uid: str, detect: KeyCondition):
        self.id = uid
        self._detect = detect

    def detect(self, state_dict: Mapping[str, object]) -> bool:
        return self._detect(state_dict)

    @abstractmethod
    def load(self, state_dict: Mapping[str, object]) -> T:
        raise NotImplementedError

    def _enhance_model(self, model: T, in_channels: int, out_channels: int, upscale: int, name) -> T:
        model.parameters_info = ModelMetadata(name=name, in_channels=in_channels, out_channels=out_channels, upscale=upscale)
        return model
