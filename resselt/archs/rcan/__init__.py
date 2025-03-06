import math
from typing import Mapping

from .arch import RCAN
from resselt.utils import get_seq_len, get_pixelshuffle_params
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture


class RCANArch(Architecture[RCAN]):
    def __init__(self):
        super().__init__(
            id='RCAN',
            detect=KeyCondition.has_any(
                KeyCondition.has_all(
                    'head.0.weight',
                    'tail.1.weight',
                    'body.0.body.0.body.0.weight',
                    'body.0.body.0.body.3.conv_du.0.weight',
                ),
                KeyCondition.has_all(
                    'head.1.weight',
                    'tail.1.weight',
                    'body.0.body.0.body.0.weight',
                    'body.0.body.0.body.3.conv_du.0.weight',
                ),
            ),
        )

    def load(self, state_dict: Mapping[str, object]) -> WrappedModel:
        n_resgroups = get_seq_len(state_dict, 'body') - 1
        n_resblocks = get_seq_len(state_dict, 'body.0.body') - 1
        head_index = 0
        scale, n_feats = get_pixelshuffle_params(state_dict, 'tail.0')
        unshuffle_mod = get_seq_len(state_dict, 'head') > 1
        n_colors = state_dict['tail.1.weight'].shape[0]
        if unshuffle_mod:
            head_index += 1
            unshuffled_channels = state_dict[f'head.{head_index}.weight'].shape[1]
            downscale_factor = int(math.sqrt(unshuffled_channels / n_colors))
            scale = 4 // downscale_factor
        rgb_range = 255  # undetectable
        kernel_size = state_dict[f'head.{head_index}.weight'].shape[-1]
        norm = 'sub_mean.weight' in state_dict
        reduction = n_feats // state_dict['body.0.body.0.body.3.conv_du.0.weight'].shape[0]
        res_scale = 1  # undetectable
        act_mode = 'relu'  # undetectable
        model = RCAN(
            scale=scale,
            n_resgroups=n_resgroups,
            n_resblocks=n_resblocks,
            n_colors=n_colors,
            rgb_range=rgb_range,
            norm=norm,
            kernel_size=kernel_size,
            n_feats=n_feats,
            reduction=reduction,
            res_scale=res_scale,
            act_mode=act_mode,
            unshuffle_mod=unshuffle_mod,
        )

        return WrappedModel(model=model, in_channels=n_colors, out_channels=n_colors, upscale=scale, name='RCAN')
