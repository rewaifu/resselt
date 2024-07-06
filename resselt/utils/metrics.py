from functools import reduce
from typing import Tuple

import numpy as np
import torch

from resselt.registry import WrappedModel


def calculate_memory_usage(
    wrapped_model: WrappedModel,
    img_size: Tuple[int, int] = (128, 128),
) -> Tuple[float, float]:
    if not torch.cuda.is_available():
        raise "CUDA is not available"

    initial_memory = torch.cuda.max_memory_allocated(device=None)

    x = torch.rand(1, wrapped_model.in_channels, *img_size)

    wrapped_model.eval()
    wrapped_model(x)

    out_memory = torch.cuda.max_memory_allocated(device=None)
    memory_per_element = (out_memory - initial_memory) / calculate_image_elements(x)

    del x

    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
    max_elements = total_gpu_memory / memory_per_element

    return max_elements, memory_per_element


def calculate_image_elements(x: torch.Tensor | np.ndarray) -> int:
    return reduce(lambda x, y: x * y, x.shape)
