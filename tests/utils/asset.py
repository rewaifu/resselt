from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import requests
import torch
from pepeline import read

from resselt.registry.architecture import WrappedModel, Architecture
from resselt.utils import canonicalize_state_dict

ROOT_DIR = Path('tests/assets')
MODEL_DIR = Path('models/')
IMAGE_DIR = Path('images/')


class AssetType(Enum):
    MODEL = 'model'
    IMAGE = 'image'


@dataclass
class Asset(ABC):
    type: AssetType
    filename: str
    url: str

    @property
    def path(self):
        return (ROOT_DIR / MODEL_DIR / self.filename) if self.type == AssetType.MODEL else (ROOT_DIR / IMAGE_DIR / self.filename)

    def _download_file(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(self.url, stream=True)
        response.raise_for_status()

        with open(self.path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f'File downloaded successfully: {self.path}')


class ModelAsset(Asset):
    def __init__(self, filename: str, url: str):
        super().__init__(type=AssetType.MODEL, filename=filename, url=url)

    def load_wrapped_model(self, architecture: Architecture) -> WrappedModel:
        if not self.path.exists():
            self._download_file()

        state_dict = torch.load(self.path, map_location='cpu')
        state_dict = canonicalize_state_dict(state_dict)

        return architecture.load(state_dict)


class ImageAsset(Asset):
    def __init__(self, filename: str, url: str):
        super().__init__(type=AssetType.IMAGE, filename=filename, url=url)

    def get_file(self) -> np.ndarray:
        if not self.path.exists():
            self._download_file()

        return read(str(self.path), None, 0)


class ImageAssets(Enum):
    MANGA_GRAY_1200_1669 = ImageAsset('manga_1200_1669.jpg', 'https://public.yor.ovh/FP02_JKtaimabu1_003.jpg')
    COLOR_CAT_120_113 = ImageAsset('cat_120_113.jpg', 'https://public.yor.ovh/cat.4362.jpg')


__all__ = ['ImageAsset', 'ModelAsset', 'ImageAssets', 'ROOT_DIR']
