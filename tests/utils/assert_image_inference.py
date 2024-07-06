import os
import sys
from pathlib import Path

import numpy as np
import torch
from pepeline import read, save
from syrupy.filters import props

from resselt.registry import WrappedModel
from resselt.utils import image2tensor, tensor2image
from .asset import ImageAssets, ROOT_DIR

IMAGE_OUTPUT_DIR = Path('output/')

disallowed_props = props('model', 'state_dict', 'device', 'dtype')


def assert_image_inference(
    wrapped_model: WrappedModel,
    model_filename: str,
    *img_assets: ImageAssets,
):
    update_mode = '--snapshot-update' in sys.argv

    wrapped_model.eval()

    for img_asset in img_assets:
        input_img = img_asset.value.get_file()
        tensor = image2tensor(input_img).unsqueeze(0)

        in_channels, in_height, in_weight = tensor.shape[1:]
        assert (
            in_channels == wrapped_model.in_channels
        ), f'Expected the input image to have {wrapped_model.in_channels} channels, got {in_channels}'

        try:
            with torch.no_grad():
                tensor = wrapped_model(tensor)
        except Exception as e:
            raise AssertionError(f'Failed on {img_asset.value.filename}: {e}') from e

        out_channels, out_height, out_weight = tensor.shape[1:]
        assert (
            out_channels == wrapped_model.out_channels
        ), f'Expected the output tensor to have {wrapped_model.out_channels} channels, got {out_channels}'

        output_img = tensor2image(tensor)

        filename, _ = os.path.splitext(img_asset.value.filename)
        model_filename, _ = os.path.splitext(model_filename)

        expected_path = ROOT_DIR / IMAGE_OUTPUT_DIR / filename / f'{model_filename}.png'

        if update_mode and not expected_path.exists():
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            save(output_img, str(expected_path))
            continue

        assert expected_path.exists(), 'Expected output image does not exist.'

        expected_img = read(str(expected_path), None, 0)

        if wrapped_model.in_channels == 1:
            close_enough = np.allclose(output_img, expected_img[:, :, 0], atol=1)
        else:
            close_enough = np.allclose(output_img, expected_img, atol=1)

        if update_mode and not close_enough:
            save(output_img, str(expected_path))
            continue

        assert close_enough, f'Failed on {img_asset.value.filename}'
