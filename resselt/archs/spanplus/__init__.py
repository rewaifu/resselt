from typing import Mapping

from .arch import SpanPlus
from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import get_seq_len, pixelshuffle_scale, dysample_scale


class SpanPlusArch(Architecture[SpanPlus]):
    def __init__(self):
        super().__init__(
            uid='spanplus',
            detect=KeyCondition.has_all('feats.0.eval_conv.weight'),
        )

    def load(self, state_dict: Mapping[str, object]):
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

        return self._enhance_model(model=model, in_channels=num_in_ch, out_channels=num_out_ch, upscale=upscale, name='SPANPlus')
