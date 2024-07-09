from __future__ import annotations

import math
from abc import abstractmethod, ABC
from typing import List, NamedTuple, Union
import numpy as np
from typing_extensions import override

from resselt.registry.architecture import WrappedModel
from resselt.utils.metrics import calculate_image_elements, calculate_memory_usage


class Size(NamedTuple):
    height: int
    width: int


class Tiler(ABC):
    def __init__(self, size: Union[Size, int]):
        self.size = Size(size, size) if isinstance(size, int) else size

    def __call__(self, image: np.ndarray) -> List[np.ndarray]:
        height, width = image.shape[:2]
        tiles = []

        self.tiling_hk(image)

        num_tiles_height = (height + self.size.height - 1) // self.size.height
        num_tiles_width = (width + self.size.width - 1) // self.size.width

        for tile_y, tile_x in np.ndindex(num_tiles_height, num_tiles_width):
            start_y = tile_y * self.size.height
            end_y = min(start_y + self.size.height, height)
            start_x = tile_x * self.size.width
            end_x = min(start_x + self.size.width, width)

            padded_tile = self._pad_tile(image[start_y:end_y, start_x:end_x])
            tiles.append(padded_tile)

        return tiles

    @abstractmethod
    def decrease_size(self):
        pass

    def tiling_hk(self, img: np.ndarray):
        pass

    def _pad_tile(self, tile: np.ndarray) -> np.ndarray:
        height, width = tile.shape[:2]
        pad_height = max(self.size.height - height, 0)
        pad_width = max(self.size.width - width, 0)

        if pad_height > 0 or pad_width > 0:
            pad_widths = ((0, pad_height), (0, pad_width)) + ((0, 0),) * (tile.ndim - 2)
            return np.pad(tile, pad_widths, mode='constant', constant_values=1)
        else:
            return tile

    def merge(
        self,
        tiles: List[np.ndarray],
        original_height: int,
        original_width: int,
        scale: int | None = None,
    ) -> np.ndarray:
        """Concatenate tiles into a single image, removing padding if present."""

        tile_size = self.size
        if scale is not None:
            tile_size = Size(tile_size.height * scale, tile_size.width * scale)
            original_height = original_height * scale
            original_width = original_width * scale

        num_tiles_height = (original_height + tile_size.height - 1) // tile_size.height
        num_tiles_width = (original_width + tile_size.width - 1) // tile_size.width

        sample_tile = tiles[0]
        if sample_tile.ndim > 2:
            img = np.zeros((original_height, original_width, sample_tile.shape[2]), dtype=sample_tile.dtype)
        else:
            img = np.zeros((original_height, original_width), dtype=sample_tile.dtype)

        for tile_index, (tile_y, tile_x) in enumerate(np.ndindex(num_tiles_height, num_tiles_width)):
            start_y = tile_y * tile_size.height
            end_y = min(start_y + tile_size.height, original_height)
            start_x = tile_x * tile_size.width
            end_x = min(start_x + tile_size.width, original_width)

            img[start_y:end_y, start_x:end_x] = tiles[tile_index][: end_y - start_y, : end_x - start_x]

        return img


class MaxTiler(Tiler):
    def __init__(self):
        super().__init__(Size(0, 0))

    @override
    def tiling_hk(self, img: np.ndarray):
        height, width = img.shape[:2]
        biggest_size = max(height, width)
        self.size = Size(biggest_size, biggest_size)

    def decrease_size(self):
        self.size = Size(max(16, self.size.height // 2), max(16, self.size.width // 2))


class ExactTiler(Tiler):
    def __init__(self, tile_size: int = 256):
        super().__init__(Size(tile_size, tile_size))

    def decrease_size(self):
        raise 'ExactTiler does not support decreasing size.'


class AutoTiler(Tiler):
    def __init__(self, wrapped_model: WrappedModel):
        super().__init__(Size(0, 0))

        max_elements, _ = calculate_memory_usage(wrapped_model)
        self.max_elements = max_elements

    @override
    def tiling_hk(self, img: np.ndarray):
        img_elements = calculate_image_elements(img)
        total_tiles = math.ceil(img_elements / self.max_elements)

        height, width = img.shape[:2]
        self.size = Size(height // total_tiles, width // total_tiles)

    def decrease_size(self):
        raise 'AutoTiler does not support decreasing size.'
