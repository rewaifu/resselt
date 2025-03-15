import math
from typing import Mapping

from .arch import RTMoSR
from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import get_seq_len


class RTMoSRArch(Architecture[RTMoSR]):
    def __init__(self):
        super().__init__(
            uid='RTMoSR',
            detect=KeyCondition.has_all(
                'body.0.norm.scale',
                'body.0.norm.offset',
                'body.0.fc1.alpha',
                'body.0.fc1.conv1.k0',
                'body.0.fc1.conv1.b0',
                'body.0.fc1.conv1.k1',
                'body.0.fc1.conv1.b1',
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
                'body.0.conv.0.poll.1.alpha',
                'body.0.conv.0.poll.1.conv1.k0',
                'body.0.conv.0.poll.1.conv1.b0',
                'body.0.conv.0.poll.1.conv1.k1',
                'body.0.conv.0.poll.1.conv1.b1',
                'body.0.conv.0.poll.1.conv2.weight',
                'body.0.conv.0.poll.1.conv2.bias',
                'body.0.conv.0.poll.1.conv3.sk.weight',
                'body.0.conv.0.poll.1.conv3.sk.bias',
                'body.0.conv.0.poll.1.conv3.conv.0.weight',
                'body.0.conv.0.poll.1.conv3.conv.0.bias',
                'body.0.conv.0.poll.1.conv3.conv.1.weight',
                'body.0.conv.0.poll.1.conv3.conv.1.bias',
                'body.0.conv.0.poll.1.conv3.conv.2.weight',
                'body.0.conv.0.poll.1.conv3.conv.2.bias',
                'body.0.conv.0.poll.1.conv3.eval_conv.weight',
                'body.0.conv.0.poll.1.conv3.eval_conv.bias',
                'body.0.conv.0.poll.1.conv_3x3_rep.weight',
                'body.0.conv.0.poll.1.conv_3x3_rep.bias',
                'body.0.conv.1.alpha1',
                'body.0.conv.1.alpha2',
                'body.0.conv.1.alpha3',
                'body.0.conv.1.alpha4',
                'body.0.conv.1.conv1x1.weight',
                'body.0.conv.1.conv1x1.bias',
                'body.0.conv.1.conv3x3.weight',
                'body.0.conv.1.conv3x3.bias',
                'body.0.conv.1.conv5x5.weight',
                'body.0.conv.1.conv5x5.bias',
                'body.0.conv.1.conv5x5_reparam.weight',
                'body.0.conv.1.conv5x5_reparam.bias',
                'to_img.0.alpha',
                'to_img.0.conv1.k0',
                'to_img.0.conv1.b0',
                'to_img.0.conv1.k1',
                'to_img.0.conv1.b1',
                'to_img.0.conv2.weight',
                'to_img.0.conv2.bias',
                'to_img.0.conv3.sk.weight',
                'to_img.0.conv3.sk.bias',
                'to_img.0.conv3.conv.0.weight',
                'to_img.0.conv3.conv.0.bias',
                'to_img.0.conv3.conv.1.weight',
                'to_img.0.conv3.conv.1.bias',
                'to_img.0.conv3.conv.2.weight',
                'to_img.0.conv3.conv.2.bias',
                'to_img.0.conv3.eval_conv.weight',
                'to_img.0.conv3.eval_conv.bias',
                'to_img.0.conv_3x3_rep.weight',
                'to_img.0.conv_3x3_rep.bias',
            ),
        )

    def load(self, state: Mapping[str, object]):
        unshuffle = False
        if 'to_feat.1.alpha' in state:
            unshuffle = True
            scale = math.isqrt(state['to_feat.1.conv_3x3_rep.weight'].shape[1] // 3)
            dim = state['to_feat.1.conv_3x3_rep.weight'].shape[0]
        else:
            scale = math.isqrt(state['to_img.0.conv_3x3_rep.weight'].shape[0] // 3)
            dim = state['to_feat.conv_3x3_rep.weight'].shape[0]
        dccm = 'body.0.fc2.alpha' in state
        se = 'body.0.conv.2.squeezing.0.weight' in state
        ffn = state['body.0.fc1.conv_3x3_rep.weight'].shape[0] / dim / 2
        n_blocks = get_seq_len(state, 'body')

        model = RTMoSR(scale=scale, dim=dim, ffn_expansion=ffn, n_blocks=n_blocks, unshuffle_mod=unshuffle, dccm=dccm, se=se)

        return self._enhance_model(model=model, in_channels=3, out_channels=3, upscale=int(2), name='RTMoSR')
