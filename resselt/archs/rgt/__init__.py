import math
from typing import Mapping

from .arch import RGT
from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import get_seq_len, get_pixelshuffle_params


def _get_split_size(state_dict: Mapping[str, object]) -> tuple[int, int]:
    # split_size can only be uniquely determined if split_size[0] == split_size[1]
    # From now on, we call ssw=split_size[0] and ssh=split_size[1]
    # What we know:
    #   a = ssw * ssh
    #   b = (ssw * 2 - 1) * (ssh * 2 - 1)
    # Since this is not enough to uniquely determine ssw and ssh, we assume:
    #   ssw <= ssh
    a = state_dict['layers.0.blocks.0.attn.attns.0.relative_position_index'].shape[0]
    b = state_dict['layers.0.blocks.0.attn.attns.0.rpe_biases'].shape[0]

    def is_solution(ssw: int, ssh: int) -> bool:
        return ssw * ssh == a and (2 * ssw - 1) * (2 * ssh - 1) == b

    # this simplifies a lot if ssw == ssh
    square_size = math.isqrt(a)
    if is_solution(square_size, square_size):
        return square_size, square_size

    # otherwise, we'll just check powers of 2
    for i in range(1, 10):
        ssw = 2**i
        for j in range(i + 1, 10):
            ssh = 2**j
            if is_solution(ssw, ssh):
                return ssw, ssh

    raise ValueError(f'No valid split_size found for {a=} and {b=}')


class RGTArch(Architecture[RGT]):
    def __init__(self):
        super().__init__(
            uid='RGT',
            detect=KeyCondition.has_all(
                'conv_first.weight',
                'before_RG.1.weight',
                'layers.0.blocks.0.gamma',
                'layers.0.blocks.0.norm1.weight',
                'layers.0.blocks.0.attn.qkv.weight',
                'layers.0.blocks.0.attn.proj.weight',
                'layers.0.blocks.0.attn.attns.0.rpe_biases',
                'layers.0.blocks.0.attn.attns.0.relative_position_index',
                'layers.0.blocks.0.attn.attns.0.pos.pos_proj.weight',
                'layers.0.blocks.0.mlp.fc1.weight',
                'layers.0.blocks.0.mlp.fc2.weight',
                'layers.0.blocks.0.norm2.weight',
                'norm.weight',
                KeyCondition.has_any(
                    # 1conv
                    'conv_after_body.weight',
                    # 3conv
                    'conv_after_body.0.weight',
                ),
                'conv_before_upsample.0.weight',
                'conv_last.weight',
            ),
        )

    def load(self, state_dict: Mapping[str, object]):
        img_size = 64  # unused
        qk_scale = None  # cannot be deduced from state_dict
        drop_rate = 0.0  # cannot be deduced from state_dict
        attn_drop_rate = 0.0  # cannot be deduced from state_dict
        drop_path_rate = 0.1  # cannot be deduced from state_dict
        img_range = 1.0  # cannot be deduced from state_dict
        c_ratio = 0.5

        in_chans = state_dict['conv_first.weight'].shape[1]
        embed_dim = state_dict['conv_first.weight'].shape[0]

        num_layers = get_seq_len(state_dict, 'layers')
        depth = [0] * num_layers
        num_heads = [2] * num_layers
        for i in range(num_layers):
            depth[i] = get_seq_len(state_dict, f'layers.{i}.blocks')
            heads_half = state_dict[f'layers.{i}.blocks.0.attn.attns.0.pos.pos3.2.weight'].shape[0]
            if embed_dim % (heads_half * 2) == 0:
                num_heads[i] = heads_half * 2
            else:
                num_heads[i] = heads_half * 2 + 1

        qkv_bias = 'layers.0.blocks.0.attn.qkv.bias' in state_dict

        mlp_ratio = state_dict['layers.0.blocks.0.mlp.fc1.weight'].shape[0] / state_dict['layers.0.blocks.0.mlp.fc1.weight'].shape[1]

        if 'conv_after_body.weight' in state_dict:
            resi_connection = '1conv'
        else:
            resi_connection = '3conv'

        # c_ratio is only defined if at least one depth is >= 2
        for i, d in enumerate(depth):
            if d >= 2:
                c_ratio = state_dict[f'layers.{i}.blocks.1.attn.conv.weight'].shape[0] / state_dict[f'layers.{i}.blocks.1.attn.conv.weight'].shape[1]
                break

        upscale, _ = get_pixelshuffle_params(state_dict, 'upsample')

        split_size = _get_split_size(state_dict)

        model = RGT(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            upscale=upscale,
            img_range=img_range,
            resi_connection=resi_connection,
            split_size=split_size,
            c_ratio=c_ratio,
        )
        return self._enhance_model(model=model, in_channels=in_chans, out_channels=in_chans, upscale=upscale, name='RGT')
