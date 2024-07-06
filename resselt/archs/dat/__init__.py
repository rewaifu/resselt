import math
from typing import Mapping

from .arch import DAT
from resselt.utils import get_seq_len, pixelshuffle_scale
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture


class DatArch(Architecture[DAT]):
    def __init__(self):
        super().__init__(
            id='dat',
            detect=KeyCondition.has_all(
                'conv_first.weight',
                'before_RG.1.weight',
                'before_RG.1.bias',
                'layers.0.blocks.0.norm1.weight',
                'layers.0.blocks.0.norm2.weight',
                'layers.0.blocks.0.ffn.fc1.weight',
                'layers.0.blocks.0.ffn.sg.norm.weight',
                'layers.0.blocks.0.ffn.sg.conv.weight',
                'layers.0.blocks.0.ffn.fc2.weight',
                'layers.0.blocks.0.attn.qkv.weight',
                'layers.0.blocks.0.attn.proj.weight',
                'layers.0.blocks.0.attn.dwconv.0.weight',
                'layers.0.blocks.0.attn.dwconv.1.running_mean',
                'layers.0.blocks.0.attn.channel_interaction.1.weight',
                'layers.0.blocks.0.attn.channel_interaction.2.running_mean',
                'layers.0.blocks.0.attn.channel_interaction.4.weight',
                'layers.0.blocks.0.attn.spatial_interaction.0.weight',
                'layers.0.blocks.0.attn.spatial_interaction.1.running_mean',
                'layers.0.blocks.0.attn.spatial_interaction.3.weight',
                'layers.0.blocks.0.attn.attns.0.rpe_biases',
                'layers.0.blocks.0.attn.attns.0.relative_position_index',
                'layers.0.blocks.0.attn.attns.0.pos.pos_proj.weight',
                'layers.0.blocks.0.attn.attns.0.pos.pos1.0.weight',
                'layers.0.blocks.0.attn.attns.0.pos.pos3.0.weight',
                'norm.weight',
            ),
        )

    def load(self, state_dict: Mapping[str, object]) -> WrappedModel:
        img_size = 64  # cannot be deduced from state dict in general
        split_size = [2, 4]
        upscale = 2

        in_chans = state_dict['conv_first.weight'].shape[1]
        embed_dim = state_dict['conv_first.weight'].shape[0]

        # num_layers = len(depth)
        num_layers = get_seq_len(state_dict, 'layers')
        depth = [get_seq_len(state_dict, f'layers.{i}.blocks') for i in range(num_layers)]

        # num_heads is linked to depth
        num_heads = [2] * num_layers
        for i in range(num_layers):
            if depth[i] >= 2:
                # that's the easy path, we can directly read the head count
                num_heads[i] = state_dict[f'layers.{i}.blocks.1.attn.temperature'].shape[0]
            else:
                # because of a head_num // 2, we can only reconstruct even head counts
                key = f'layers.{i}.blocks.0.attn.attns.0.pos.pos3.2.weight'
                num_heads[i] = state_dict[key].shape[0] * 2

        upsampler = 'pixelshuffle' if 'conv_last.weight' in state_dict else 'pixelshuffledirect'
        resi_connection = '1conv' if 'conv_after_body.weight' in state_dict else '3conv'

        if upsampler == 'pixelshuffle':
            upscale = 1
            for i in range(0, get_seq_len(state_dict, 'upsample'), 2):
                num_feat = state_dict[f'upsample.{i}.weight'].shape[1]
                shape = state_dict[f'upsample.{i}.weight'].shape[0]
                upscale *= int(math.sqrt(shape // num_feat))
        elif upsampler == 'pixelshuffledirect':
            num_feat = state_dict['upsample.0.weight'].shape[1]
            upscale = pixelshuffle_scale(state_dict['upsample.0.weight'].shape[0], in_chans)

        qkv_bias = 'layers.0.blocks.0.attn.qkv.bias' in state_dict

        expansion_factor = float(state_dict['layers.0.blocks.0.ffn.fc1.weight'].shape[0] / embed_dim)

        if 'layers.0.blocks.2.attn.attn_mask_0' in state_dict:
            attn_mask_0_x, attn_mask_0_y, _attn_mask_0_z = state_dict['layers.0.blocks.2.attn.attn_mask_0'].shape

            img_size = int(math.sqrt(attn_mask_0_x * attn_mask_0_y))

        if 'layers.0.blocks.0.attn.attns.0.rpe_biases' in state_dict:
            split_sizes = state_dict['layers.0.blocks.0.attn.attns.0.rpe_biases'][-1] + 1
            split_size = [int(x) for x in split_sizes]

        model = DAT(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            split_size=split_size,
            depth=depth,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            qkv_bias=qkv_bias,
            upscale=upscale,
            resi_connection=resi_connection,
            upsampler=upsampler,
        )

        return WrappedModel(model=model, in_channels=in_chans, out_channels=in_chans, upscale=upscale, name='DAT')
