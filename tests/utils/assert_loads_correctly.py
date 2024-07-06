from inspect import getsource
from typing import TypeVar, Callable

import torch

from resselt.registry import Architecture

T = TypeVar('T', bound=torch.nn.Module)


def assert_loads_correctly(
    architecture: Architecture,
    *models: Callable[[], T],
):
    for model_fn in models:
        model_name = getsource(model_fn)

        try:
            model = model_fn()
        except Exception as e:
            raise AssertionError(f'Failed to create model: {model_name}: {e}') from e

        try:
            state_dict = model.state_dict()
            loaded = architecture.load(state_dict)
        except Exception as e:
            raise AssertionError(f'Failed to load: {model_name}: {e}') from e

        assert type(loaded.model) is type(model), f'Expected {model_name} to be loaded correctly, but found a {type(loaded.model)} instead.'

        original_keys = set(model.state_dict().keys())
        loaded_keys = set(loaded.model.state_dict().keys())

        if original_keys != loaded_keys:
            missing_keys = original_keys - loaded_keys
            extra_keys = loaded_keys - original_keys
            raise AssertionError(f'Key mismatch in {model_name}. Missing keys: {missing_keys}, Extra keys: {extra_keys}')

        for key in original_keys:
            original_param = model.state_dict()[key]
            loaded_param = loaded.model.state_dict()[key]
            if original_param.shape != loaded_param.shape:
                raise AssertionError(
                    f'Parameter {key} shapes differ in {model_name}: original {original_param.shape}, loaded {loaded_param.shape}'
                )
