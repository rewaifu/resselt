from typing import Mapping

from .arch import GateRV2
from ...factory import KeyCondition, Architecture
from ...utilities.state_dict import get_seq_len


class GateRV2Arch(Architecture[GateRV2]):
    def __init__(self):
        super().__init__(
            uid='GateRv2',
            detect=KeyCondition.has_all(
                'in_to_dim.weight',
                'in_to_dim.bias',
                'encode.0.gated.0.gamma0',
                'encode.0.gated.0.gamma1',
                'encode.0.gated.0.local.0.scale',
                'encode.0.gated.0.local.0.offset',
                'encode.0.gated.0.local.1.weight',
                'encode.0.gated.0.local.1.bias',
                'encode.0.gated.0.local.2.weight',
                'encode.0.gated.0.local.2.bias',
                'encode.0.gated.0.sca.1.weight',
                'encode.0.gated.0.sca.1.bias',
                'encode.0.gated.0.glob.norm.scale',
                'encode.0.gated.0.glob.norm.offset',
                'encode.0.gated.0.glob.fc1.weight',
                'encode.0.gated.0.glob.fc1.bias',
                'encode.0.gated.0.glob.token_mix.dwconv_hw.weight',
                'encode.0.gated.0.glob.token_mix.dwconv_hw.bias',
                'encode.0.gated.0.glob.token_mix.dwconv_w.weight',
                'encode.0.gated.0.glob.token_mix.dwconv_w.bias',
                'encode.0.gated.0.glob.token_mix.dwconv_h.weight',
                'encode.0.gated.0.glob.token_mix.dwconv_h.bias',
                'encode.0.gated.0.glob.fc2.weight',
                'encode.0.gated.0.glob.fc2.bias',
                'encode.0.scale.0.weight',
                'encode.1.gated.0.gamma0',
                'encode.1.gated.0.gamma1',
                'encode.1.gated.0.local.0.scale',
                'encode.1.gated.0.local.0.offset',
                'encode.1.gated.0.local.1.weight',
                'encode.1.gated.0.local.1.bias',
                'encode.1.gated.0.local.2.weight',
                'encode.1.gated.0.local.2.bias',
                'encode.1.gated.0.sca.1.weight',
                'encode.1.gated.0.sca.1.bias',
                'encode.1.gated.0.glob.norm.scale',
                'encode.1.gated.0.glob.norm.offset',
                'encode.1.gated.0.glob.fc1.weight',
                'encode.1.gated.0.glob.fc1.bias',
                'encode.1.gated.0.glob.token_mix.dwconv_hw.weight',
                'encode.1.gated.0.glob.token_mix.dwconv_hw.bias',
                'encode.1.gated.0.glob.token_mix.dwconv_w.weight',
                'encode.1.gated.0.glob.token_mix.dwconv_w.bias',
                'encode.1.gated.0.glob.token_mix.dwconv_h.weight',
                'encode.1.gated.0.glob.token_mix.dwconv_h.bias',
                'encode.1.gated.0.glob.fc2.weight',
                'encode.1.gated.0.glob.fc2.bias',
                'encode.1.scale.0.weight',
                'latent.0.norm.scale',
                'latent.0.norm.offset',
                'latent.0.fc1.weight',
                'latent.0.fc1.bias',
                'latent.0.token_mix.query_conv.weight',
                'latent.0.token_mix.query_conv.bias',
                'latent.0.token_mix.key_conv.weight',
                'latent.0.token_mix.key_conv.bias',
                'latent.0.token_mix.value_conv.weight',
                'latent.0.token_mix.value_conv.bias',
                'latent.0.fc2.weight',
                'latent.0.fc2.bias',
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
                'decode.1.scale.0.weight',
                'decode.1.gated.0.gamma0',
                'decode.1.gated.0.gamma1',
                'decode.1.gated.0.local.0.scale',
                'decode.1.gated.0.local.0.offset',
                'decode.1.gated.0.local.1.weight',
                'decode.1.gated.0.local.1.bias',
                'decode.1.gated.0.local.2.weight',
                'decode.1.gated.0.local.2.bias',
                'decode.1.gated.0.sca.1.weight',
                'decode.1.gated.0.sca.1.bias',
                'decode.1.gated.0.glob.norm.scale',
                'decode.1.gated.0.glob.norm.offset',
                'decode.1.gated.0.glob.fc1.weight',
                'decode.1.gated.0.glob.fc1.bias',
                'decode.1.gated.0.glob.token_mix.dwconv_hw.weight',
                'decode.1.gated.0.glob.token_mix.dwconv_hw.bias',
                'decode.1.gated.0.glob.token_mix.dwconv_w.weight',
                'decode.1.gated.0.glob.token_mix.dwconv_w.bias',
                'decode.1.gated.0.glob.token_mix.dwconv_h.weight',
                'decode.1.gated.0.glob.token_mix.dwconv_h.bias',
                'decode.1.gated.0.glob.fc2.weight',
                'decode.1.gated.0.glob.fc2.bias',
                'decode.1.shor.weight',
                'decode.1.shor.bias',
            ),
        )

    def load(self, state: Mapping[str, object]):
        dim, in_ch = state['in_to_dim.weight'].shape[:2]
        enc_blocks = [get_seq_len(state, f'encode.{i}.gated') for i in range(get_seq_len(state, 'encode'))]
        latent = get_seq_len(state, 'latent')
        dec_blocks = [get_seq_len(state, f'decode.{i}.gated') for i in range(get_seq_len(state, 'decode'))]
        if 'upsample.MetaUpsample' in state:
            upsample = ['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample']
            _, index, scale, _, out_ch, upsample_dim, _ = [value.item() for value in state['to_img.MetaUpsample']]
            upsampler = upsample[int(index)]
        else:
            scale, upsample_dim, upsampler = 1, 32, 'conv'
        model = GateRV2(in_ch, dim, enc_blocks, dec_blocks, latent, scale, upsampler, upsample_dim)

        return self._enhance_model(model=model, in_channels=in_ch, out_channels=int(in_ch), upscale=scale, name='GateRv2')
