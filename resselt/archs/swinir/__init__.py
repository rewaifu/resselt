import math
from typing import Mapping

from .arch import SwinIR
from ...factory import KeyCondition, Architecture
from ...utilities.state_dict import get_seq_len, get_pixelshuffle_params


class SwinIRArch(Architecture[SwinIR]):
    def __init__(self):
        super().__init__(
            uid='SwinIR',
            detect=KeyCondition.has_all(
                'layers.0.residual_group.blocks.0.norm1.weight',
                'conv_first.weight',
                'layers.0.residual_group.blocks.0.mlp.fc1.bias',
                'layers.0.residual_group.blocks.0.attn.relative_position_index',
            ),
        )

    def load(self, state_dict: Mapping[str, object]):
        # Defaults
        img_size = 64
        patch_size = 1
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.1
        ape = False
        patch_norm = True
        use_checkpoint = False
        start_unshuffle = 1

        if 'conv_before_upsample.0.weight' in state_dict:
            if 'conv_up1.weight' in state_dict:
                upsampler = 'nearest+conv'
            else:
                upsampler = 'pixelshuffle'
        elif 'upsample.0.weight' in state_dict:
            upsampler = 'pixelshuffledirect'
        else:
            upsampler = ''

        if 'conv_first.1.weight' in state_dict:
            state_dict['conv_first.weight'] = state_dict.pop('conv_first.1.weight')
            state_dict['conv_first.bias'] = state_dict.pop('conv_first.1.bias')
            start_unshuffle = round(math.sqrt(state_dict['conv_first.weight'].shape[1] // 3))

        num_in_ch = state_dict['conv_first.weight'].shape[1]
        if 'conv_last.weight' in state_dict:
            num_out_ch = state_dict['conv_last.weight'].shape[0]
        else:
            num_out_ch = num_in_ch

        upscale = 1
        if upsampler == 'nearest+conv':
            upsample_keys = [x for x in state_dict if 'conv_up' in x and 'bias' not in x]

            for _upsample_key in upsample_keys:
                upscale *= 2
        elif upsampler == 'pixelshuffle':
            upscale, num_feat = get_pixelshuffle_params(state_dict, 'upsample')
        elif upsampler == 'pixelshuffledirect':
            upscale = int(math.sqrt(state_dict['upsample.0.bias'].shape[0] // num_out_ch))

        embed_dim = state_dict['conv_first.weight'].shape[0]

        mlp_ratio = float(state_dict['layers.0.residual_group.blocks.0.mlp.fc1.bias'].shape[0] / embed_dim)

        window_size = int(math.sqrt(state_dict['layers.0.residual_group.blocks.0.attn.relative_position_index'].shape[0]))

        if 'layers.0.residual_group.blocks.1.attn_mask' in state_dict:
            img_size = int(math.sqrt(state_dict['layers.0.residual_group.blocks.1.attn_mask'].shape[0]) * window_size)

        # depths & num_heads
        num_layers = get_seq_len(state_dict, 'layers')
        depths = []
        num_heads = []
        for i in range(num_layers):
            depths.append(get_seq_len(state_dict, f'layers.{i}.residual_group.blocks'))
            num_heads.append(state_dict[f'layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table'].shape[1])

        if 'conv_after_body.weight' in state_dict:
            resi_connection = '1conv'
        else:
            resi_connection = '3conv'

        # The JPEG models are the only ones with window-size 7, and they also use this range
        img_range = 255.0 if window_size == 7 else 1.0

        in_nc = num_in_ch // start_unshuffle**2
        out_nc = num_out_ch

        model = SwinIR(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_nc,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            upscale=upscale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
            start_unshuffle=start_unshuffle,
        )

        return self._enhance_model(model=model, in_channels=in_nc, out_channels=out_nc, upscale=upscale, name='SwinIR')
