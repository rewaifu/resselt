from enum import Enum
import numpy as np
import torch
from torch import Tensor


def empty_cuda_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


class UpscaleDType(Enum):
    F32 = torch.float32
    BF16 = torch.bfloat16
    F16 = torch.half


def image2tensor(
    value: list[np.ndarray] | np.ndarray,
    out_type: torch.dtype = torch.float32,
) -> list[Tensor] | Tensor:
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        if len(img.shape) == 2:
            tensor = torch.from_numpy(img[None, ...])
        else:
            tensor = torch.from_numpy(img.transpose(2, 0, 1))

        if tensor.dtype != out_type:
            tensor = tensor.to(out_type)

        return tensor

    if isinstance(value, list):
        return [_to_tensor(i) for i in value]
    else:
        return _to_tensor(value)


def tensor2image(
    value: list[torch.Tensor] | torch.Tensor,
    out_type=np.float32,
) -> list[np.ndarray] | np.ndarray:
    def _to_ndarray(tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.squeeze(0).detach().cpu()

        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        img = tensor.numpy().transpose(1, 2, 0)
        if out_type == np.uint8:
            img = (img * 255.0).round()

        return img.astype(out_type)

    if isinstance(value, list):
        return [_to_ndarray(i) for i in value]
    else:
        return _to_ndarray(value)
