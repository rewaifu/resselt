from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Mapping

import torch

from resselt.registry.key_condition import KeyCondition

T = TypeVar('T', bound=torch.nn.Module, covariant=True)


class Architecture(ABC, Generic[T]):
    def __init__(self, id: str, detect: KeyCondition):
        self.id = id
        self._detect = detect

    def detect(self, state_dict: Mapping[str, object]) -> bool:
        return self._detect(state_dict)

    @abstractmethod
    def load(self, state_dict: Mapping[str, object]) -> WrappedModel:
        raise NotImplementedError


class WrappedModel(Generic[T]):
    def __init__(self, model: T, in_channels: int, out_channels: int, upscale: int, name: str = 'unknown name'):
        self.model = model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale = upscale
        self.name = name

    def to(self, device: torch.device | torch.dtype):
        self.model.to(device)
        return self

    def parameters(self) -> (str, int, int):
        """
        return name, in_ch, out_ch
        """
        return self.name, self.in_channels, self.out_channels

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        strict: bool = True,
        assign: bool = False,
    ):
        self.model.load_state_dict(state_dict, strict, assign)
        return self

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


__all__ = ['Architecture', 'WrappedModel']
