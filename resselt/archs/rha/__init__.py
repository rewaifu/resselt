from typing import Mapping

from .arch import RHA
from resselt.utils import get_seq_len
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture


class RHAArch(Architecture[RHA]):
    def __init__(self):
        super().__init__(
            id='RHA',
            detect=KeyCondition.has_all(
                'body.0.down_sample',
                'body.0.body.0.norm.weight',
                'body.0.body.0.norm.bias',
                'body.0.body.0.fc1.weight',
                'body.0.body.0.fc1.bias',
                'body.0.body.0.conv.att.1.alpha1',
                'body.0.body.0.conv.att.1.alpha2',
                'body.0.body.0.conv.att.1.alpha3',
                'body.0.body.0.conv.att.1.alpha4',
                'body.0.body.0.conv.att.1.conv1x1.weight',
                'body.0.body.0.conv.att.1.conv1x1.bias',
                'body.0.body.0.conv.att.1.conv3x3.weight',
                'body.0.body.0.conv.att.1.conv3x3.bias',
                'body.0.body.0.conv.att.1.conv5x5.weight',
                'body.0.body.0.conv.att.1.conv5x5.bias',
                'body.0.body.0.conv.att.1.conv5x5_reparam.weight',
                'body.0.body.0.conv.att.1.conv5x5_reparam.bias',
                'body.0.body.0.conv.conv.alpha1',
                'body.0.body.0.conv.conv.alpha2',
                'body.0.body.0.conv.conv.alpha3',
                'body.0.body.0.conv.conv.alpha4',
                'body.0.body.0.conv.conv.conv1x1.weight',
                'body.0.body.0.conv.conv.conv1x1.bias',
                'body.0.body.0.conv.conv.conv3x3.weight',
                'body.0.body.0.conv.conv.conv3x3.bias',
                'body.0.body.0.conv.conv.conv5x5.weight',
                'body.0.body.0.conv.conv.conv5x5.bias',
                'body.0.body.0.conv.conv.conv5x5_reparam.weight',
                'body.0.body.0.conv.conv.conv5x5_reparam.bias',
                'body.0.body.0.conv.aggr.0.weight',
                'body.0.body.0.conv.aggr.0.bias',
                'body.0.body.0.fc2.weight',
                'body.0.body.0.fc2.bias',
                'to_img.MetaUpsample',
            ),
        )

    def load(self, state: Mapping[str, object]) -> WrappedModel:
        upsample = ['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample']
        unshuffle = 1
        unshuffle_mod = False
        if 'unshuffle' in state:
            unshuffle = state['unshuffle'].item()
            unshuffle_mod = True
            dim, in_ch, _, _ = state['to_feat.1.weight'].shape
            in_ch //= unshuffle**2
        else:
            dim, in_ch, _, _ = state['to_feat.weight'].shape

        group_blocks = get_seq_len(state, 'body')
        res_blocks = get_seq_len(state, 'body.0.body') - 1
        down_list = [state[f'body.{index}.down_sample'].item() for index in range(group_blocks)]
        expansion_ratio = state['body.0.body.0.fc1.weight'].shape[0] / 2 / dim
        _, index, scale, _, out_ch, upsample_dim, _ = [value.item() for value in state['to_img.MetaUpsample']]
        upsampler = upsample[int(index)]
        scale //= unshuffle

        model = RHA(
            dim=dim,
            scale=scale,
            in_ch=in_ch,
            out_ch=out_ch,
            mid_dim=upsample_dim,
            down_list=down_list,
            expansion_ratio=expansion_ratio,
            group_blocks=group_blocks,
            res_blocks=res_blocks,
            upsample=upsampler,
            unshuffle_mod=unshuffle_mod,
        )

        return WrappedModel(model=model, in_channels=in_ch, out_channels=out_ch, upscale=scale, name='RHA')
