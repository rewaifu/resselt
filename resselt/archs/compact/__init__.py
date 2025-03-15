from typing import Mapping

from .arch import SRVGGNetCompact
from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import get_seq_len, pixelshuffle_scale


class CompactArch(Architecture[SRVGGNetCompact]):
    def __init__(self):
        super().__init__(
            uid='Compact',
            detect=KeyCondition.has_all(
                'body.0.weight',
                'body.1.weight',
            ),
        )

    def load(self, state_dict: Mapping[str, object]):
        state = state_dict

        highest_num = get_seq_len(state, 'body') - 1

        in_nc = state['body.0.weight'].shape[1]
        num_feat = state['body.0.weight'].shape[0]
        num_conv = (highest_num - 2) // 2

        pixelshuffle_shape = state[f'body.{highest_num}.bias'].shape[0]

        scale = pixelshuffle_scale(pixelshuffle_shape, in_nc)
        model = SRVGGNetCompact(
            num_in_ch=in_nc,
            num_out_ch=in_nc,
            num_feat=num_feat,
            num_conv=num_conv,
            upscale=scale,
        )

        return self._enhance_model(model=model, in_channels=in_nc, out_channels=in_nc, upscale=scale, name='Compact')
