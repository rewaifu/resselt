from typing import Mapping

from .arch import mosr as MoSR
from resselt.utils import get_seq_len, pixelshuffle_scale, dysample_scale
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture


class MoSRArch(Architecture[MoSR]):
    def __init__(self):
        super().__init__(
            id='MoSR',
            detect=KeyCondition.has_all(
                'in_to_dim.weight',
                'in_to_dim.bias',
                'gblocks.0.gated.0.dccm.0.weight',
            ),
        )

    def load(self, state: Mapping[str, object]) -> WrappedModel:
        len_blocks = get_seq_len(state, 'gblocks')
        blocks = []
        for i in range(len_blocks):
            blocks.append(get_seq_len(state, 'gblocks.0.gated'))
        dim = state['in_to_dim.weight'].shape[0]
        in_ch = state['in_to_dim.weight'].shape[1]
        expansion = state['gblocks.0.gated.0.gcnn.fc2.weight'].shape
        expansion_ratio = expansion[1] / expansion[0]
        if 'upsampler.weight' in state:
            upsampler = 'conv'
            upscale = 1
            out_ch = state['upsampler.weight'].shape[0]
        elif 'upsampler.0.weight' in state:
            upsampler = 'ps'
            out_ch = in_ch
            upscale = pixelshuffle_scale(state['upsampler.0.weight'].shape[0], out_ch)
        else:
            upsampler = 'dys'
            out_ch = state['upsampler.end_conv.weight'].shape[0]
            upscale = dysample_scale(state['upsampler.offset.weight'].shape[0])
        model = MoSR(in_ch=in_ch, out_ch=out_ch, upscale=upscale, blocks=blocks, dim=dim, upsampler=upsampler, expansion_ratio=expansion_ratio)

        return WrappedModel(model=model, in_channels=in_ch, out_channels=out_ch, upscale=upscale, name='MoSR')
