import numpy as np
import torch

from resselt.registry.architecture import WrappedModel
from .tensor import image2tensor, empty_cuda_cache, tensor2image, UpscaleDType
from .tiler import Tiler


def upscale_with_tiler(
    img: np.ndarray, tiler: Tiler, wrapped_model: WrappedModel, device: torch.device, upscale_type: UpscaleDType = UpscaleDType.F32
) -> np.ndarray:
    def _upscale() -> np.ndarray:
        tiles = tiler(img)
        amp = upscale_type != UpscaleDType.F32
        for tile in tiles:
            try:
                tile_tensor = image2tensor(tile.img).to(device).unsqueeze(0)
                with torch.autocast(device_type=str(device), dtype=upscale_type.value, enabled=amp):
                    with torch.inference_mode():
                        output_tensor = wrapped_model(tile_tensor)
                tile.img = tensor2image(output_tensor)
            except torch.cuda.OutOfMemoryError:
                tiler.decrease_size()
                empty_cuda_cache()
                return _upscale()
            except RuntimeError as e:
                print(f'RuntimeError: {e}')
                raise e

        height, width = img.shape[:2]
        result = tiler.merge(tiles, height, width, wrapped_model.upscale)

        return result

    wrapped_model.to(device).eval()

    output_img = _upscale()

    return output_img
