from typing import Mapping

from .arch import MoESR
from resselt.utils import get_seq_len
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture


class MoESRArch(Architecture[MoESR]):
    def __init__(self):
        super().__init__(
            id='MoESR',
            detect=KeyCondition.has_all(
                'gamma',
                'metadata',
                'in_to_dim.weight',
                'in_to_dim.bias',
                'blocks.0.gamma',
                'blocks.0.blocks.0.gamma',
                'blocks.0.blocks.0.token_mix.gamma',
                'blocks.0.blocks.0.token_mix.norm.weight',
                'blocks.0.blocks.0.token_mix.norm.bias',
                'blocks.0.blocks.0.token_mix.fc1.weight',
                'blocks.0.blocks.0.token_mix.fc1.bias',
                'blocks.0.blocks.0.token_mix.conv.weight',
                'blocks.0.blocks.0.token_mix.conv.bias',
                'blocks.0.blocks.0.token_mix.fc2.weight',
                'blocks.0.blocks.0.token_mix.fc2.bias',
                'blocks.0.blocks.0.ffn.0.weight',
                'blocks.0.blocks.0.ffn.0.bias',
                'blocks.0.blocks.0.ffn.1.project_in.weight',
                'blocks.0.blocks.0.ffn.1.project_in.bias',
                'blocks.0.blocks.0.ffn.1.dwconv.weight',
                'blocks.0.blocks.0.ffn.1.dwconv.bias',
                'blocks.0.blocks.0.ffn.1.project_out.weight',
                'blocks.0.blocks.0.ffn.1.project_out.bias',
                'blocks.0.blocks.0.se.squeezing.0.weight',
                'blocks.0.blocks.0.se.squeezing.0.bias',
            ),
        )

    def load(self, state: Mapping[str, object]) -> WrappedModel:
        expansion_factor_shape = state['blocks.0.blocks.0.ffn.1.project_in.weight'].shape

        upsampler = 'conv'
        upsample_dim = 64
        metadata = state['metadata']
        in_ch = metadata[0].item()
        out_ch = metadata[1].item()
        scale = metadata[2].item()
        dim = state['in_to_dim.weight'].shape[0]
        n_blocks = get_seq_len(state, 'blocks')
        n_block = get_seq_len(state, 'blocks.0.blocks') - 1
        expansion_factor = expansion_factor_shape[0] / expansion_factor_shape[1]
        expansion_esa_shape = state[f'blocks.0.blocks.{n_block}.conv1.weight'].shape
        expansion_esa = expansion_esa_shape[0] / expansion_factor_shape[1]
        if 'upscale.weight' in state:
            upsampler = 'conv'
        elif 'upscale.init_pos' in state:
            upsampler = 'dys'
        elif 'upscale.0.weight' in state:
            if 'upscale.3.weight' in state:
                upsampler = 'n+c'
            elif 'upscale.2.weight' in state:
                upsampler = 'ps'
                upsample_dim = state['upscale.0.weight'].shape[0]
            else:
                upsampler = 'psd'

        model = MoESR(
            in_ch=in_ch,
            out_ch=out_ch,
            scale=scale,
            n_blocks=n_blocks,
            n_block=n_block,
            dim=dim,
            expansion_factor=expansion_factor,
            expansion_esa=expansion_esa,
            upsampler=upsampler,
            upsample_dim=upsample_dim,
        )

        return WrappedModel(model=model, in_channels=in_ch, out_channels=out_ch, upscale=scale, name='MoESR')
