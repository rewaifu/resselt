import math
from typing import Mapping

from .arch import RealTimeMoSR
from resselt.utils import get_seq_len
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture


class RTMoSRArch(Architecture[RealTimeMoSR]):
    def __init__(self):
        super().__init__(
            id='RTMoSR',
            detect=KeyCondition.has_all(
                'body.0.norm.weight',
                'body.0.norm.bias',
                'body.0.fc1.alpha',
                'body.0.fc1.conv1.weight',
                'body.0.fc1.conv1.bias',
                'body.0.fc1.conv2.weight',
                'body.0.fc1.conv2.bias',
                'body.0.fc1.conv3.sk.weight',
                'body.0.fc1.conv3.sk.bias',
                'body.0.fc1.conv3.conv.0.weight',
                'body.0.fc1.conv3.conv.0.bias',
                'body.0.fc1.conv3.conv.1.weight',
                'body.0.fc1.conv3.conv.1.bias',
                'body.0.fc1.conv3.conv.2.weight',
                'body.0.fc1.conv3.conv.2.bias',
                'body.0.fc1.conv3.eval_conv.weight',
                'body.0.fc1.conv3.eval_conv.bias',
                'body.0.fc1.conv_3x3_rep.weight',
                'body.0.fc1.conv_3x3_rep.bias',
                'body.0.conv.dwconv_hw.weight',
                'body.0.conv.dwconv_hw.bias',
                'body.0.conv.dwconv_w.weight',
                'body.0.conv.dwconv_w.bias',
                'body.0.conv.dwconv_h.weight',
                'body.0.conv.dwconv_h.bias',
                'body.0.fc2.weight',
                'body.0.fc2.bias',
                'to_img.0.weight',
                'to_img.0.bias',
            ),
        )

    def load(self, state: Mapping[str, object]) -> WrappedModel:
        unshuffle = False
        if 'to_feat.1.weight' in state:
            unshuffle = True
            scale = math.isqrt(state['to_feat.1.weight'].shape[1] // 3)
            dim = state['to_feat.1.weight'].shape[0]
        else:
            scale = math.isqrt(state['to_img.0.weight'].shape[0] // 3)
            dim = state['to_feat.weight'].shape[0]
        ffn = state['body.0.fc1.conv1.weight'].shape[0] / dim / 2
        n_blocks = get_seq_len(state, 'body')

        model = RealTimeMoSR(scale=scale, dim=dim, ffn_expansion=ffn, n_blocks=n_blocks, unshuffle_mod=unshuffle)

        return WrappedModel(model=model, in_channels=3, out_channels=3, upscale=int(scale), name='RTMoSR')
