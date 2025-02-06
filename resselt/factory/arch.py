import torch
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Mapping

from .key_condition import KeyCondition

T = TypeVar('T', bound=torch.nn.Module, covariant=True)


class ModelMetadata:
    """Mixin for adding SR model specific attributes and methods"""

    in_channels: int
    out_channels: int
    upscale: int
    name: str

    def parameters_info(self) -> tuple[str, int, int, int]:
        """Return name, in_channels, out_channels"""
        return self.name, self.in_channels, self.out_channels, self.upscale


class Architecture(ABC, Generic[T]):
    def __init__(self, id: str, detect: KeyCondition):
        self.id = id
        self._detect = detect

    def detect(self, state_dict: Mapping[str, object]) -> bool:
        return self._detect(state_dict)

    @abstractmethod
    def load(self, state_dict: Mapping[str, object]) -> T:
        raise NotImplementedError

    def _enhance_model(self, model: T, in_channels: int, out_channels: int, upscale: int, name) -> T:
        model.in_channels = in_channels
        model.out_channels = out_channels
        model.upscale = upscale
        model.name = name

        model.parameters_info = ModelMetadata.parameters_info.__get__(model)

        return model
