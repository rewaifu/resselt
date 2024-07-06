from typing import Mapping

from .arch import SpanPlus
from resselt.utils import get_seq_len, dysample_scale, pixelshuffle_scale
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture


class SpanPlusArch(Architecture[SpanPlus]):
    def __init__(self):
        super().__init__(
            id='spanplus',
            detect=KeyCondition.has_all(
                'feats.0.eval_conv.weight',
            ),
        )

    def load(self, state_dict: Mapping[str, object]) -> WrappedModel:
        n_feats = get_seq_len(state_dict, 'feats') - 1
        blocks = [get_seq_len(state_dict, f'feats.{n_feat + 1}.block_n') for n_feat in range(n_feats)]
        num_in_ch = state_dict['feats.0.eval_conv.weight'].shape[1]
        feature_channels = state_dict['feats.0.eval_conv.weight'].shape[0]
        if 'upsampler.0.weight' in state_dict.keys():
            upsampler = 'ps'
            num_out_ch = num_in_ch
            upscale = pixelshuffle_scale(state_dict['upsampler.0.weight'].shape[0], num_out_ch)
        else:
            upsampler = 'dys'
            num_out_ch = state_dict['upsampler.end_conv.weight'].shape[0]
            upscale = dysample_scale(state_dict['upsampler.offset.weight'].shape[0])

        model = SpanPlus(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            blocks=blocks,
            feature_channels=feature_channels,
            upscale=upscale,
            upsampler=upsampler,
        )

        return WrappedModel(model=model, in_channels=num_in_ch, out_channels=num_out_ch, upscale=upscale, name='SPANPlus')
