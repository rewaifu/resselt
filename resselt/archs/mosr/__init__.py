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
                'gblocks.0.weight',
                'gblocks.0.bias',
                'gblocks.1.norm.weight',
                'gblocks.1.norm.bias',
                'gblocks.1.fc1.weight',
                'gblocks.1.fc1.bias',
                'gblocks.1.conv.weight',
                'gblocks.1.conv.bias',
                'gblocks.1.fc2.weight',
                'gblocks.1.fc2.bias',
            ),
        )

    def load(self, state: Mapping[str, object]) -> WrappedModel:
        # Get values from state
        n_block = get_seq_len(state, 'gblocks') - 6
        in_ch = state['gblocks.0.weight'].shape[1]
        dim = state['gblocks.0.weight'].shape[0]

        # Calculate expansion ratio and convolution ratio
        expansion_ratio = (state['gblocks.1.fc1.weight'].shape[0] / state['gblocks.1.fc1.weight'].shape[1]) / 2
        conv_ratio = state['gblocks.1.conv.weight'].shape[0] / dim
        kernel_size = state['gblocks.1.conv.weight'].shape[2]
        # Determine upsampler type and calculate upscale
        if 'upsampler.init_pos' in state:
            upsampler = 'dys'
            out_ch = state['upsampler.end_conv.weight'].shape[0]
            upscale = dysample_scale(state['upsampler.offset.weight'].shape[0])
        else:
            upsampler = 'ps'
            out_ch = in_ch
            upscale = pixelshuffle_scale(state['upsampler.0.weight'].shape[0], out_ch)

        model = MoSR(
            in_ch=in_ch,
            out_ch=out_ch,
            n_block=n_block,
            upscale=upscale,
            dim=dim,
            upsampler=upsampler,
            expansion_ratio=expansion_ratio,
            conv_ratio=conv_ratio,
            kernel_size=kernel_size,
        )

        return WrappedModel(model=model, in_channels=in_ch, out_channels=out_ch, upscale=upscale, name='MoSR')
