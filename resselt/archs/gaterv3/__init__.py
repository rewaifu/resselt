from typing import Mapping

from .arch import GateRV3
from ...factory import KeyCondition, Architecture
from ...utilities.state_dict import get_seq_len


class GateRV3Arch(Architecture[GateRV3]):
    def __init__(self):
        super().__init__(
            uid='GateRV3',
            detect=KeyCondition.has_all(
                'in_to_dim.weight',
                'in_to_dim.bias',
                'gater_encode.0.gated.0.gamma0',
                'gater_encode.0.gated.0.gamma1',
                'gater_encode.0.gated.0.local.0.scale',
                'gater_encode.0.gated.0.local.0.offset',
                'gater_encode.0.gated.0.local.1.weight',
                'gater_encode.0.gated.0.local.1.bias',
                'gater_encode.0.gated.0.local.2.weight',
                'gater_encode.0.gated.0.local.2.bias',
                'gater_encode.0.gated.0.sca.1.weight',
                'gater_encode.0.gated.0.sca.1.bias',
                'gater_encode.0.gated.0.glob.norm.scale',
                'gater_encode.0.gated.0.glob.norm.offset',
                'gater_encode.0.gated.0.glob.fc1.weight',
                'gater_encode.0.gated.0.glob.fc1.bias',
                'gater_encode.0.gated.0.glob.token_mix.dwconv_hw.weight',
                'gater_encode.0.gated.0.glob.token_mix.dwconv_hw.bias',
                'gater_encode.0.gated.0.glob.token_mix.dwconv_w.weight',
                'gater_encode.0.gated.0.glob.token_mix.dwconv_w.bias',
                'gater_encode.0.gated.0.glob.token_mix.dwconv_h.weight',
                'gater_encode.0.gated.0.glob.token_mix.dwconv_h.bias',
                'gater_encode.0.gated.0.glob.fc2.weight',
                'gater_encode.0.gated.0.glob.fc2.bias',
                'gater_encode.0.scale.0.weight',
                'span_block0.c1_r.sk.weight',
                'span_block0.c1_r.conv.0.weight',
                'span_block0.c1_r.conv.1.weight',
                'span_block0.c1_r.conv.2.weight',
                'span_block0.c1_r.eval_conv.weight',
                'span_block0.c2_r.sk.weight',
                'span_block0.c2_r.conv.0.weight',
                'span_block0.c2_r.conv.1.weight',
                'span_block0.c2_r.conv.2.weight',
                'span_block0.c2_r.eval_conv.weight',
                'span_block0.c3_r.sk.weight',
                'span_block0.c3_r.conv.0.weight',
                'span_block0.c3_r.conv.1.weight',
                'span_block0.c3_r.conv.2.weight',
                'span_block0.c3_r.eval_conv.weight',
                'span_n_b.0.c1_r.sk.weight',
                'span_n_b.0.c1_r.conv.0.weight',
                'span_n_b.0.c1_r.conv.1.weight',
                'span_n_b.0.c1_r.conv.2.weight',
                'span_n_b.0.c1_r.eval_conv.weight',
                'span_n_b.0.c2_r.sk.weight',
                'span_n_b.0.c2_r.conv.0.weight',
                'span_n_b.0.c2_r.conv.1.weight',
                'span_n_b.0.c2_r.conv.2.weight',
                'span_n_b.0.c2_r.eval_conv.weight',
                'span_n_b.0.c3_r.sk.weight',
                'span_n_b.0.c3_r.conv.0.weight',
                'span_n_b.0.c3_r.conv.1.weight',
                'span_n_b.0.c3_r.conv.2.weight',
                'span_n_b.0.c3_r.eval_conv.weight',
                'span_end.c1_r.sk.weight',
                'span_end.c1_r.conv.0.weight',
                'span_end.c1_r.conv.1.weight',
                'span_end.c1_r.conv.2.weight',
                'span_end.c1_r.eval_conv.weight',
                'span_end.c2_r.sk.weight',
                'span_end.c2_r.conv.0.weight',
                'span_end.c2_r.conv.1.weight',
                'span_end.c2_r.conv.2.weight',
                'span_end.c2_r.eval_conv.weight',
                'span_end.c3_r.sk.weight',
                'span_end.c3_r.conv.0.weight',
                'span_end.c3_r.conv.1.weight',
                'span_end.c3_r.conv.2.weight',
                'span_end.c3_r.eval_conv.weight',
                'sisr_end_conv.sk.weight',
                'sisr_end_conv.sk.bias',
                'sisr_end_conv.conv.0.weight',
                'sisr_end_conv.conv.0.bias',
                'sisr_end_conv.conv.1.weight',
                'sisr_end_conv.conv.1.bias',
                'sisr_end_conv.conv.2.weight',
                'sisr_end_conv.conv.2.bias',
                'sisr_end_conv.eval_conv.weight',
                'sisr_end_conv.eval_conv.bias',
                'sisr_cat_conv.weight',
                'sisr_cat_conv.bias',
                'decode.0.scale.0.weight',
                'decode.0.gated.0.gamma0',
                'decode.0.gated.0.gamma1',
                'decode.0.gated.0.local.0.scale',
                'decode.0.gated.0.local.0.offset',
                'decode.0.gated.0.local.1.weight',
                'decode.0.gated.0.local.1.bias',
                'decode.0.gated.0.local.2.weight',
                'decode.0.gated.0.local.2.bias',
                'decode.0.gated.0.sca.1.weight',
                'decode.0.gated.0.sca.1.bias',
                'decode.0.gated.0.glob.norm.scale',
                'decode.0.gated.0.glob.norm.offset',
                'decode.0.gated.0.glob.fc1.weight',
                'decode.0.gated.0.glob.fc1.bias',
                'decode.0.gated.0.glob.token_mix.dwconv_hw.weight',
                'decode.0.gated.0.glob.token_mix.dwconv_hw.bias',
                'decode.0.gated.0.glob.token_mix.dwconv_w.weight',
                'decode.0.gated.0.glob.token_mix.dwconv_w.bias',
                'decode.0.gated.0.glob.token_mix.dwconv_h.weight',
                'decode.0.gated.0.glob.token_mix.dwconv_h.bias',
                'decode.0.gated.0.glob.fc2.weight',
                'decode.0.gated.0.glob.fc2.bias',
                'decode.0.shor.weight',
                'decode.0.shor.bias',
            ),
        )

    def load(self, state: Mapping[str, object]):
        dim, in_ch = state['in_to_dim.weight'].shape[:2]
        enc_blocks = [get_seq_len(state, f'gater_encode.{i}.gated') for i in range(get_seq_len(state, 'gater_encode'))]
        latent = get_seq_len(state, 'latent')
        dec_blocks = [get_seq_len(state, f'decode.{i}.gated') for i in range(get_seq_len(state, 'decode'))]
        if 'dim_to_in.MetaUpsample' in state:
            upsample = ['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample']
            _, index, scale, _, out_ch, upsample_dim, _ = [value.item() for value in state['dim_to_in.MetaUpsample']]
            upsampler = upsample[int(index)]
        else:
            scale, upsample_dim, upsampler = 1, 32, 'conv'
        attention = 'latent.0.token_mix.qkv_dwconv.weight' in state
        span_blocks = get_seq_len(state, 'span_n_b')
        model = GateRV3(in_ch, dim, enc_blocks, dec_blocks, latent, scale, upsampler, upsample_dim, attention=attention, span_blocks=span_blocks)

        return self._enhance_model(model=model, in_channels=in_ch, out_channels=int(in_ch), upscale=scale, name='GateRV3')
