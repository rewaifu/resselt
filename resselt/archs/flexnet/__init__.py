import math
from typing import Mapping

from .arch import FlexNet
from ...factory import KeyCondition, Architecture
from ...utilities.state_dict import get_seq_len


class FlexNetArch(Architecture[FlexNet]):
    def __init__(self):
        super().__init__(
            uid='FlexNet',
            detect=KeyCondition.has_all(
                'short_cut.block.0.weight',
                'short_cut.block.0.bias',
                'short_cut.block.2.weight',
                'short_cut.block.2.bias',
                'short_cut.conv11.weight',
                'short_cut.conv11.bias',
                'in_to_feat.weight',
                'in_to_feat.bias',
                KeyCondition.has_any(
                    'pipeline.enc0.0.t_blocks.0.gamma1',
                    'pipeline.att.0.t_blocks.0.gamma1',
                ),
            ),
        )

    def load(self, state: Mapping[str, object]):
        window_size = int(state['window_size'])
        dim, inp_channels = state['in_to_feat.weight'].shape[:2]
        out_channels = inp_channels

        pipeline_type = 'meta' if 'pipeline.enc0.0.t_blocks.0.gamma1' in state else 'linear'
        if pipeline_type == 'meta':
            num_blocks = [get_seq_len(state, f'pipeline.enc{index}.0.t_blocks') for index in range(4)]
            hidden_rate_shape = state['pipeline.enc0.0.t_blocks.0.ffn.key.weight'].shape
            channel_norm = 'pipeline.enc0.0.t_blocks.0.ffn.key_norm.weight' in state

        else:
            numbet = get_seq_len(state, 'pipeline.att')
            num_blocks = [get_seq_len(state, f'pipeline.att.{index}.t_blocks') for index in range(numbet)]
            hidden_rate_shape = state['pipeline.att.0.t_blocks.2.ffn.key.weight'].shape
            channel_norm = 'pipeline.att.0.t_blocks.0.ffn.key_norm.weight' in state
        hidden_rate = hidden_rate_shape[0] // hidden_rate_shape[1]
        if 'to_img.1.0.weight' in state:
            upsampler = 'n+c'
            scale = int(state['scale_factor'])
            end_index = get_seq_len(state, 'to_img.1') - 1
            out_channels = state[f'to_img.1.{end_index}.weight'].shape[0]
        elif 'to_img.init_pos' in state:
            upsampler = 'dys'
            out_channels = state['to_img.end_conv.weight'].shape[0]
            scale = math.isqrt(state['to_img.offset.weight'].shape[0] // 8)
        else:
            upsampler = 'ps'
            scale = math.isqrt(state['to_img.0.weight'].shape[0] // out_channels)
        model = FlexNet(
            inp_channels=inp_channels,
            out_channels=out_channels,
            scale=scale,
            dim=dim,
            num_blocks=num_blocks,
            window_size=window_size,
            hidden_rate=hidden_rate,
            channel_norm=channel_norm,
            pipeline_type=pipeline_type,
            upsampler=upsampler,
        )

        return self._enhance_model(model=model, in_channels=inp_channels, out_channels=out_channels, upscale=scale, name='FlexNet')
