from __future__ import annotations

import math
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, NamedTuple, Union, Tuple, Optional
import numpy as np
from typing_extensions import override


class Size(NamedTuple):
    height: int
    width: int


@dataclass
class Padding:
    top: int
    right: int
    bottom: int
    left: int

    @staticmethod
    def all(value: int) -> Padding:
        return Padding(value, value, value, value)

    @staticmethod
    def to(value: Padding | int) -> Padding:
        if isinstance(value, int):
            return Padding.all(value)
        return value

    def min(self, other: Padding | int) -> Padding:
        other = Padding.to(other)
        return Padding(
            min(self.top, other.top),
            min(self.right, other.right),
            min(self.bottom, other.bottom),
            min(self.left, other.left),
        )

    @property
    def horizontal(self) -> int:
        return self.left + self.right

    @property
    def vertical(self) -> int:
        return self.top + self.bottom

    def scale(self, factor: int) -> Padding:
        return Padding(
            self.top * factor,
            self.right * factor,
            self.bottom * factor,
            self.left * factor,
        )

    def remove_from(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        return image[
            self.top : (h - self.bottom),
            self.left : (w - self.right),
            ...,
        ]


@dataclass
class Region:
    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> Size:
        return Size(self.height, self.width)

    def intersect(self, other: Region) -> Region:
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        width = min(self.x + self.width, other.x + other.width) - x
        height = min(self.y + self.height, other.y + other.height) - y
        return Region(x, y, width, height)

    def child_padding(self, child: Region) -> Padding:
        left = child.x - self.x
        top = child.y - self.y
        right = self.width - child.width - left
        bottom = self.height - child.height - top
        return Padding(top, right, bottom, left)

    def add_padding(self, pad: Padding) -> Region:
        return Region(
            x=self.x - pad.left,
            y=self.y - pad.top,
            width=self.width + pad.horizontal,
            height=self.height + pad.vertical,
        )

    def remove_padding(self, pad: Padding) -> Region:
        return self.add_padding(pad.scale(-1))

    def scale(self, factor: int) -> Region:
        return Region(
            self.x * factor,
            self.y * factor,
            self.width * factor,
            self.height * factor,
        )

    def read_from(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if (h, w) == self.size:
            return img

        return img[
            self.y : (self.y + self.height),
            self.x : (self.x + self.width),
            ...,
        ]

    def write_into(self, lhs: np.ndarray, rhs: np.ndarray):
        h, w = rhs.shape[:2]
        c = 1 if rhs.ndim == 2 else rhs.shape[2]

        assert (h, w) == self.size
        assert c == lhs.shape[2]

        if c == 1:
            if lhs.ndim == 2 and rhs.ndim == 3:
                rhs = rhs[:, :, 0]
            if lhs.ndim == 3 and rhs.ndim == 2:
                rhs = np.expand_dims(rhs, axis=2)

        lhs[
            self.y : (self.y + self.height),
            self.x : (self.x + self.width),
            ...,
        ] = rhs


@dataclass
class Tile:
    region: Region
    padding: Padding
    img: np.ndarray


class Tiler(ABC):
    def __init__(self, size: Size):
        self.size = size

    def __call__(self, img: np.ndarray) -> List[Tile]:
        h, w = img.shape[:2]
        img_region = Region(0, 0, w, h)

        self.tiling_hk(img)

        tile_count_x = math.ceil(w / self.size.width)
        tile_count_y = math.ceil(h / self.size.height)
        tile_size_x = math.ceil(w / tile_count_x)
        tile_size_y = math.ceil(h / tile_count_y)

        tiles = []

        for y, x in np.ndindex(tile_count_y, tile_count_x):
            tile = Region(x * tile_size_x, y * tile_size_y, tile_size_x, tile_size_y).intersect(img_region)
            pad = img_region.child_padding(tile).min(16)
            padded_tile = tile.add_padding(pad)

            tiles.append(Tile(padding=pad, img=padded_tile.read_from(img), region=tile))

        return tiles

    @abstractmethod
    def decrease_size(self):
        pass

    def tiling_hk(self, img: np.ndarray):
        pass

    def merge(
        self,
        tiles: List[Tile],
        original_height: int,
        original_width: int,
        factor: Optional[int] = 1,
    ) -> np.ndarray:
        """Concatenate tiles into a single image, removing padding if present."""

        sample_tile = tiles[0].img
        if sample_tile.ndim > 2:
            merge_result = np.zeros((original_height * factor, original_width * factor, sample_tile.shape[2]), dtype=sample_tile.dtype)
        else:
            merge_result = np.zeros((original_height * factor, original_width * factor), dtype=sample_tile.dtype)

        for tile in tiles:
            region = tile.region.scale(factor)
            padding = tile.padding.scale(factor)

            region.write_into(merge_result, padding.remove_from(tile.img))

        return merge_result


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
    def __init__(self, size: Union[Size, Tuple[int, int], int] = 256):
        if isinstance(size, int):
            size = Size(size, size)
        elif isinstance(size, tuple):
            size = Size(size[0], size[1])
        elif not isinstance(size, Size):
            raise TypeError('size must be an instance of Size, a tuple of two integers, or an integer')

        super().__init__(size)

    def decrease_size(self):
        raise 'ExactTiler does not support decreasing size.'


class NoTiling(Tiler):
    def __init__(self):
        super().__init__(Size(0, 0))

    @override
    def tiling_hk(self, img: np.ndarray):
        h, w = img.shape[:2]
        self.size = Size(h, w)

    def decrease_size(self):
        raise 'NoTiling does not support decreasing size.'
