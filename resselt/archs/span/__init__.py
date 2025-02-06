from typing import Mapping

from .arch import SPAN
import torch

from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import pixelshuffle_scale


class SPANArch(Architecture[SPAN]):
    def __init__(self):
        super().__init__(
            uid='SPAN',
            detect=KeyCondition.has_all(
                'conv_1.sk.weight',
                'block_1.c1_r.sk.weight',
                'block_1.c1_r.eval_conv.weight',
                'block_1.c3_r.eval_conv.weight',
                'conv_cat.weight',
                'conv_2.sk.weight',
                'conv_2.eval_conv.weight',
                'upsampler.0.weight',
            ),
        )

    def load(self, state_dict: Mapping[str, object]):
        norm = True
        img_range = 255.0  # cannot be deduced from state_dict
        rgb_mean = (0.4488, 0.4371, 0.4040)  # cannot be deduced from state_dict

        num_in_ch = state_dict['conv_1.sk.weight'].shape[1]
        feature_channels = state_dict['conv_1.sk.weight'].shape[0]
        num_out_ch = num_in_ch
        # pixelshuffel shenanigans
        upscale = pixelshuffle_scale(
            state_dict['upsampler.0.weight'].shape[0],
            num_in_ch,
        )

        # norm
        if 'no_norm' in state_dict:
            norm = False
            state_dict['no_norm'] = torch.zeros(1)

        model = SPAN(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            feature_channels=feature_channels,
            upscale=upscale,
            norm=norm,
            img_range=img_range,
            rgb_mean=rgb_mean,
        )

        return self._enhance_model(model=model, in_channels=num_in_ch, out_channels=num_out_ch, upscale=upscale, name='SPAN')
