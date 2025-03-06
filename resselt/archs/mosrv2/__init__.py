import math
from typing import Mapping

from .arch import MoSRv2
from resselt.utils import get_seq_len
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture


class MoSRv2Arch(Architecture[MoSRv2]):
    def __init__(self):
        super().__init__(
            id='MoSRv2',
            detect=KeyCondition.has_any(
                KeyCondition.has_all(
                    'gblocks.1.weight',
                    'gblocks.1.bias',
                    'gblocks.2.gamma',
                    KeyCondition.has_any(
                        KeyCondition.has_all('gblocks.2.norm.scale', 'gblocks.2.norm.offset'),
                        KeyCondition.has_all('gblocks.2.norm.weight', 'gblocks.2.norm.bias'),
                    ),
                    'gblocks.2.fc1.weight',
                    'gblocks.2.fc1.bias',
                    'gblocks.2.conv.dwconv_hw.weight',
                    'gblocks.2.conv.dwconv_hw.bias',
                    'gblocks.2.conv.dwconv_w.weight',
                    'gblocks.2.conv.dwconv_w.bias',
                    'gblocks.2.conv.dwconv_h.weight',
                    'gblocks.2.conv.dwconv_h.bias',
                    'gblocks.2.fc2.weight',
                    'gblocks.2.fc2.bias',
                    'to_img.MetaUpsample',
                    'to_img.0.weight',
                    'to_img.0.bias',
                ),
                KeyCondition.has_all(
                    'gblocks.0.weight',
                    'gblocks.0.bias',
                    'gblocks.1.gamma',
                    KeyCondition.has_any(
                        KeyCondition.has_all('gblocks.1.norm.scale', 'gblocks.1.norm.offset'),
                        KeyCondition.has_all('gblocks.1.norm.weight', 'gblocks.1.norm.bias'),
                    ),
                    'gblocks.1.fc1.weight',
                    'gblocks.1.fc1.bias',
                    'gblocks.1.conv.dwconv_hw.weight',
                    'gblocks.1.conv.dwconv_hw.bias',
                    'gblocks.1.conv.dwconv_w.weight',
                    'gblocks.1.conv.dwconv_w.bias',
                    'gblocks.1.conv.dwconv_h.weight',
                    'gblocks.1.conv.dwconv_h.bias',
                    'gblocks.1.fc2.weight',
                    'gblocks.1.fc2.bias',
                    'to_img.MetaUpsample',
                    'to_img.0.weight',
                    'to_img.0.bias',
                ),
            ),
        )

    def load(self, state: Mapping[str, object]) -> WrappedModel:
        samplemods = ['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample']
        _, upsampler, scale, dim, in_ch, mid_dim, _ = [i.item() for i in state['to_img.MetaUpsample']]
        upsampler = samplemods[upsampler]
        n_block = get_seq_len(state, 'gblocks')
        if 'gblocks.0.weight' in state:
            unshuffle_mod = False
            n_block -= 6
            expansion_ratio = state['gblocks.1.fc1.weight'].shape[0] // 2 / dim
            rms_norm = 'gblocks.1.norm.scale' in state
        else:
            scale = math.isqrt(state['gblocks.1.weight'].shape[1] // in_ch)
            n_block -= 7
            unshuffle_mod = True
            expansion_ratio = state['gblocks.2.fc1.weight'].shape[0] // 2 / dim
            rms_norm = 'gblocks.2.norm.scale' in state

        model = MoSRv2(
            in_ch=in_ch,
            scale=scale,
            n_block=n_block,
            dim=dim,
            upsampler=upsampler,
            expansion_ratio=expansion_ratio,
            mid_dim=mid_dim,
            unshuffle_mod=unshuffle_mod,
            rms_norm=rms_norm,
        )

        return WrappedModel(model=model, in_channels=in_ch, out_channels=in_ch, upscale=scale, name='MoSRv2')
