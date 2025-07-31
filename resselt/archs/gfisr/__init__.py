from typing import Mapping
from .arch import GFISR
from ...factory import KeyCondition, Architecture
from ...utilities.state_dict import get_seq_len


class GateRV3Arch(Architecture[GFISR]):
    def __init__(self):
        super().__init__(
            uid='GFISR',
            detect=KeyCondition.has_all(
                'in_to_dim.weight',
                'in_to_dim.bias',
                'net.0.gamma',
                'net.0.norm.weight',
                'net.0.norm.bias',
                'net.0.fc1.weight',
                'net.0.fc1.bias',
                'net.0.conv.dwconv_hw.weight',
                'net.0.conv.dwconv_hw.bias',
                'net.0.conv.dwconv_w.weight',
                'net.0.conv.dwconv_w.bias',
                'net.0.conv.dwconv_h.weight',
                'net.0.conv.dwconv_h.bias',
                'net.0.fc2.weight',
                'net.0.fc2.bias',
                'dim_to_out.MetaUpsample',
            ),
        )

    def load(self, state: Mapping[str, object]):
        upsample = ['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample', 'transpose+conv', 'lda', 'pa_up']
        _, index, scale, _, out_ch, upsample_dim, _ = [value.item() for value in state['dim_to_out.MetaUpsample']]
        upsampler = upsample[int(index)]

        fft_mode = 'net.0.conv.fsas.ln.weight' in state
        if 'in_to_dim.weight' in state:
            dim, in_nc = state['in_to_dim.weight'].shape[:2]
            pixel_unshuffle = False
        else:
            dim, in_nc = state['in_to_dim.1.weight'].shape[:2]
            if in_nc % 16 == 0:
                in_nc //= 16
            else:
                in_nc //= 4
            pixel_unshuffle = True
        n_blocks = get_seq_len(state, 'net')

        expansion_ratio = state['net.0.fc1.bias'].shape[0] / 2 / dim
        model = GFISR(
            in_nc=in_nc,
            dim=dim,
            expansion_ratio=expansion_ratio,
            fft_mode=fft_mode,
            scale=scale,
            out_nc=out_ch,
            upsampler=upsampler,
            upsample_dim=upsample_dim,
            pixel_unshuffle=pixel_unshuffle,
            n_blocks=n_blocks,
        )

        return self._enhance_model(model=model, in_channels=int(in_nc), out_channels=int(out_ch), upscale=scale, name='GFISR')
