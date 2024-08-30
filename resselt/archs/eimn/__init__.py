import re
from typing import Mapping

from .arch import eimn
from resselt.utils import get_seq_len, pixelshuffle_scale
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture


class eimnArch(Architecture[eimn]):
    def __init__(self):
        super().__init__(
            id='eimn',
            detect=KeyCondition.has_all(
                'head.0.weight',
                'head.0.bias',
                'tail.0.weight',
                'tail.0.bias',
                'block1.0.layer_scale_1',
                'block1.0.layer_scale_2',
                'block1.0.norm1.weight',
                'block1.0.norm1.bias',
                'block1.0.norm1.running_mean',
                'block1.0.norm1.running_var',
                'block1.0.norm1.num_batches_tracked',
                'block1.0.attn.region.weight',
                'block1.0.attn.region.bias',
                'block1.0.attn.spatial_1.weight',
                'block1.0.attn.spatial_1.bias',
                'block1.0.attn.spatial_2.weight',
                'block1.0.attn.spatial_2.bias',
                'block1.0.attn.fusion.weight',
                'block1.0.attn.fusion.bias',
                'block1.0.attn.proj_value.0.weight',
                'block1.0.attn.proj_value.0.bias',
                'block1.0.attn.proj_query.0.weight',
                'block1.0.attn.proj_query.0.bias',
                'block1.0.attn.out.weight',
                'block1.0.attn.out.bias',
                'block1.0.norm2.weight',
                'block1.0.norm2.bias',
                'block1.0.norm2.running_mean',
                'block1.0.norm2.running_var',
                'block1.0.norm2.num_batches_tracked',
                'block1.0.mlp.linear_in.weight',
                'block1.0.mlp.linear_in.bias',
                'block1.0.mlp.SAL.weight',
                'block1.0.mlp.SAL.bias',
                'block1.0.mlp.linear_out.weight',
                'block1.0.mlp.linear_out.bias',
                'block1.0.mlp.DFFM.norm.weight',
                'block1.0.mlp.DFFM.norm.bias',
                'block1.0.mlp.DFFM.global_reduce.weight',
                'block1.0.mlp.DFFM.global_reduce.bias',
                'block1.0.mlp.DFFM.local_reduce.weight',
                'block1.0.mlp.DFFM.local_reduce.bias',
                'block1.0.mlp.DFFM.channel_expand.weight',
                'block1.0.mlp.DFFM.channel_expand.bias',
                'block1.0.mlp.DFFM.spatial_expand.weight',
                'block1.0.mlp.DFFM.spatial_expand.bias',
                'norm1.weight',
                'norm1.bias',
            ),
        )

    def load(self, state: Mapping[str, object]) -> WrappedModel:
        pattern = r'block(\d+)'
        numbers = [int(re.search(pattern, s).group(1)) for s in state.keys() if re.search(pattern, s)]
        num_stages = max(numbers)
        depths = get_seq_len(state, 'block1')
        mlp_ratio_shape = state['block1.0.mlp.linear_in.weight'].shape
        mlp_ratio = mlp_ratio_shape[0] // 2 / mlp_ratio_shape[1]
        embed_dim = state['head.0.weight'].shape[0]
        scale = pixelshuffle_scale(state['tail.0.weight'].shape[0], 3)

        model = eimn(
            embed_dims=embed_dim,
            scale=scale,
            depths=depths,
            mlp_ratios=mlp_ratio,
            num_stages=num_stages,
        )

        return WrappedModel(model=model, in_channels=3, out_channels=3, upscale=scale, name='EIMN')
