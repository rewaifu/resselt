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
                'in_to_dim.weight',
                'in_to_dim.bias',
                'blocks.0.blocks.0.gamma',
                'blocks.0.blocks.0.norm.weight',
                'blocks.0.blocks.0.norm.bias',
                'blocks.0.blocks.0.fc1.weight',
                'blocks.0.blocks.0.fc1.bias',
                'blocks.0.blocks.0.conv.dwconv_hw.weight',
                'blocks.0.blocks.0.conv.dwconv_hw.bias',
                'blocks.0.blocks.0.conv.dwconv_w.weight',
                'blocks.0.blocks.0.conv.dwconv_w.bias',
                'blocks.0.blocks.0.conv.dwconv_h.weight',
                'blocks.0.blocks.0.conv.dwconv_h.bias',
                'blocks.0.blocks.0.fc2.weight',
                'blocks.0.blocks.0.fc2.bias',
                'upscale.MetaUpsample',
            ),
        )

    def load(self, state: Mapping[str, object]) -> WrappedModel:
        upsample = ['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample']
        dim, in_ch = state['in_to_dim.weight'].shape[:2]
        n_blocks = get_seq_len(state, 'blocks')
        n_block = get_seq_len(state, 'blocks.0.blocks')
        expansion_factor_shape = state['blocks.0.blocks.0.fc1.weight'].shape
        expansion_factor = (expansion_factor_shape[0] / expansion_factor_shape[1]) / 2
        expansion_msg_shape = state['blocks.0.msg.gated.0.fc1.weight'].shape
        expansion_msg = (expansion_msg_shape[0] / expansion_msg_shape[1]) / 2
        _, index, scale, _, out_ch, upsample_dim, _ = state['upscale.MetaUpsample']
        upsampler = upsample[int(index)]

        model = MoESR(
            in_ch=in_ch,
            out_ch=int(out_ch),
            scale=int(scale),
            n_blocks=n_blocks,
            n_block=n_block,
            dim=dim,
            expansion_factor=expansion_factor,
            expansion_msg=expansion_msg,
            upsampler=upsampler,
            upsample_dim=int(upsample_dim),
        )

        return WrappedModel(model=model, in_channels=in_ch, out_channels=int(out_ch), upscale=int(scale), name='MoESR')
