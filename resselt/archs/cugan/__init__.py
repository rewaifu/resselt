from typing import Mapping, Union, Literal

from .arch import UpCunet2x, UpCunet2x_fast, UpCunet3x, UpCunet4x
import torch
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture

_CUGAN = Union[UpCunet2x, UpCunet3x, UpCunet4x, UpCunet2x_fast]


class CUGANArch(Architecture[_CUGAN]):
    def __init__(self):
        super().__init__(
            id='CuGAN',
            detect=KeyCondition.has_all(
                'unet1.conv1.conv.0.weight',
                'unet1.conv1.conv.2.weight',
                'unet1.conv1_down.weight',
                'unet1.conv2.conv.0.weight',
                'unet1.conv2.conv.2.weight',
                'unet1.conv2.seblock.conv1.weight',
                'unet1.conv2_up.weight',
                'unet1.conv_bottom.weight',
                'unet2.conv1.conv.0.weight',
                'unet2.conv1_down.weight',
                'unet2.conv2.conv.0.weight',
                'unet2.conv2.seblock.conv1.weight',
                'unet2.conv3.conv.0.weight',
                'unet2.conv3.seblock.conv1.weight',
                'unet2.conv3_up.weight',
                'unet2.conv4.conv.0.weight',
                'unet2.conv4_up.weight',
                'unet2.conv5.weight',
                'unet2.conv_bottom.weight',
            ),
        )

    def load(self, state_dict: Mapping[str, object]) -> WrappedModel:
        scale: Literal[2, 3, 4]
        in_channels: int
        out_channels: int
        pro: bool = False

        tags: list[str] = []
        if 'pro' in state_dict:
            pro = True
            tags.append('pro')
            state_dict['pro'] = torch.zeros(1)

        in_channels = state_dict['unet1.conv1.conv.0.weight'].shape[1]

        if 'conv_final.weight' in state_dict and in_channels == 12:
            # UpCunet2x_fast
            scale = 2
            in_channels = 3  # hard coded in UpCunet2x_fast
            out_channels = 3  # hard coded in UpCunet2x_fast
            model = UpCunet2x_fast(in_channels=in_channels, out_channels=out_channels)
            tags.append('fast')
        elif 'conv_final.weight' in state_dict:
            # UpCunet4x
            scale = 4
            out_channels = 3  # hard coded in UpCunet4x
            model = UpCunet4x(in_channels=in_channels, out_channels=out_channels, pro=pro)
        elif state_dict['unet1.conv_bottom.weight'].shape[2] == 5:
            # UpCunet3x
            scale = 3
            out_channels = state_dict['unet2.conv_bottom.weight'].shape[0]
            model = UpCunet3x(in_channels=in_channels, out_channels=out_channels, pro=pro)
        else:
            # UpCunet2x
            scale = 2
            out_channels = state_dict['unet2.conv_bottom.weight'].shape[0]
            model = UpCunet2x(in_channels=in_channels, out_channels=out_channels, pro=pro)

        return WrappedModel(model=model, in_channels=in_channels, out_channels=out_channels, upscale=scale, name='CUGAN')
