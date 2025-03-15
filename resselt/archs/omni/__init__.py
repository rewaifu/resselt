import math
from typing import Mapping

from .arch import OmniSR
import warnings

from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import pixelshuffle_scale, get_seq_len

warnings.filterwarnings('ignore')


class OmniArch(Architecture[OmniSR]):
    def __init__(self):
        super().__init__(
            uid='OmniSR',
            detect=KeyCondition.has_all(
                'residual_layer.0.residual_layer.0.layer.0.fn.0.weight',
                'input.weight',
                'up.0.weight',
            ),
        )

    def load(self, state_dict: Mapping[str, object]):
        # Remove junk from the state dict
        state_dict_keys = set(state_dict.keys())
        for key in state_dict_keys:
            if key.endswith(('total_ops', 'total_params')):
                del state_dict[key]
        window_size = 8

        num_feat = state_dict['input.weight'].shape[0]
        num_in_ch = state_dict['input.weight'].shape[1]
        num_out_ch = num_in_ch
        bias = 'input.bias' in state_dict

        pixelshuffle_shape = state_dict['up.0.weight'].shape[0]
        up_scale = pixelshuffle_scale(pixelshuffle_shape, num_in_ch)

        res_num = get_seq_len(state_dict, 'residual_layer')
        block_num = get_seq_len(state_dict, 'residual_layer.0.residual_layer') - 1

        rel_pos_bias_key = 'residual_layer.0.residual_layer.0.layer.2.fn.rel_pos_bias.weight'
        if rel_pos_bias_key in state_dict:
            pe = True
            rel_pos_bias_weight = state_dict[rel_pos_bias_key].shape[0]
            window_size = int((math.sqrt(rel_pos_bias_weight) + 1) / 2)
        else:
            pe = False

        model = OmniSR(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat,
            block_num=block_num,
            pe=pe,
            window_size=window_size,
            res_num=res_num,
            up_scale=up_scale,
            bias=bias,
        )

        return self._enhance_model(model=model, in_channels=num_in_ch, out_channels=num_out_ch, upscale=up_scale, name='OmniSR')
