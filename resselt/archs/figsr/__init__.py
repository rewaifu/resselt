from typing import Mapping

from .arch import FIGSR
from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import get_seq_len


class FIGSRrch(Architecture[FIGSR]):
    def __init__(self):
        super().__init__(
            uid='FIGSR',
            detect=KeyCondition.has_all(
                'in_to_dim.weight',
                'in_to_dim.bias',
                'gfisr_body_half.0.norm.scale',
                'gfisr_body_half.0.norm.offset',
                'gfisr_body_half.0.norm.eps',
                'gfisr_body_half.0.norm.rms',
                'gfisr_body_half.0.fc1.weight',
                'gfisr_body_half.0.fc1.bias',
                'gfisr_body_half.0.conv.fu.rn.scale',
                'gfisr_body_half.0.conv.fu.rn.offset',
                'gfisr_body_half.0.conv.fu.rn.eps',
                'gfisr_body_half.0.conv.fu.rn.rms',
                'gfisr_body_half.0.conv.fu.post_norm.scale',
                'gfisr_body_half.0.conv.fu.post_norm.offset',
                'gfisr_body_half.0.conv.fu.post_norm.eps',
                'gfisr_body_half.0.conv.fu.post_norm.rms',
                'gfisr_body_half.0.conv.fu.fdc.weight',
                'gfisr_body_half.0.conv.fu.fdc.bias',
                'gfisr_body_half.0.conv.fu.fpe.weight',
                'gfisr_body_half.0.conv.fu.fpe.bias',
                'gfisr_body_half.0.conv.convhw.weight',
                'gfisr_body_half.0.conv.convhw.bias',
                'gfisr_body_half.0.conv.convw.weight',
                'gfisr_body_half.0.conv.convw.bias',
                'gfisr_body_half.0.conv.convh.weight',
                'gfisr_body_half.0.conv.convh.bias',
                'gfisr_body_half.0.fc2.weight',
                'gfisr_body_half.0.fc2.bias',
                'gfisr_body_half_2.0.norm.scale',
                'gfisr_body_half_2.0.norm.offset',
                'gfisr_body_half_2.0.norm.eps',
                'gfisr_body_half_2.0.norm.rms',
                'gfisr_body_half_2.0.fc1.weight',
                'gfisr_body_half_2.0.fc1.bias',
                'gfisr_body_half_2.0.conv.fu.rn.scale',
                'gfisr_body_half_2.0.conv.fu.rn.offset',
                'gfisr_body_half_2.0.conv.fu.rn.eps',
                'gfisr_body_half_2.0.conv.fu.rn.rms',
                'gfisr_body_half_2.0.conv.fu.post_norm.scale',
                'gfisr_body_half_2.0.conv.fu.post_norm.offset',
                'gfisr_body_half_2.0.conv.fu.post_norm.eps',
                'gfisr_body_half_2.0.conv.fu.post_norm.rms',
                'gfisr_body_half_2.0.conv.fu.fdc.weight',
                'gfisr_body_half_2.0.conv.fu.fdc.bias',
                'gfisr_body_half_2.0.conv.fu.fpe.weight',
                'gfisr_body_half_2.0.conv.fu.fpe.bias',
                'gfisr_body_half_2.0.conv.convhw.weight',
                'gfisr_body_half_2.0.conv.convhw.bias',
                'gfisr_body_half_2.0.conv.convw.weight',
                'gfisr_body_half_2.0.conv.convw.bias',
                'gfisr_body_half_2.0.conv.convh.weight',
                'gfisr_body_half_2.0.conv.convh.bias',
                'gfisr_body_half_2.0.fc2.weight',
                'gfisr_body_half_2.0.fc2.bias',
                'cat_to_dim.weight',
                'cat_to_dim.bias',
                'upscale.MetaUpsample',
            ),
        )

    def load(self, state_dict: Mapping[str, object]):
        state = state_dict
        sample_mods = [
            'conv',
            'pixelshuffledirect',
            'pixelshuffle',
            'nearest+conv',
            'dysample',
            'transpose+conv',
            'lda',
            'pa_up',
        ]

        _, upsampler, scale, dim, out_nc, mid_dim, _ = [i.item() for i in state['upscale.MetaUpsample']]
        upsampler = sample_mods[upsampler]
        dim, in_nc = state['in_to_dim.weight'].shape[:2]
        expansion_ratio = state['gfisr_body_half.0.fc1.weight'].shape[0] / 2 / dim
        n_blocks = get_seq_len(state, 'gfisr_body_half')
        n_blocks += get_seq_len(state, 'gfisr_body_half_2') - 1
        gc = state['gfisr_body_half.0.conv.convh.bias'].shape[0]
        square_kernel_size = state['gfisr_body_half.0.conv.convhw.weight'].shape[2]
        band_kernel_size = state['gfisr_body_half.0.conv.convh.weight'].shape[2]
        model = FIGSR(
            in_nc=in_nc,
            dim=dim,
            expansion_ratio=expansion_ratio,
            scale=scale,
            out_nc=out_nc,
            upsampler=upsampler,
            mid_dim=mid_dim,
            n_blocks=n_blocks,
            gc=gc,
            square_kernel_size=square_kernel_size,
            band_kernel_size=band_kernel_size,
        )

        return self._enhance_model(model=model, in_channels=in_nc, out_channels=in_nc, upscale=scale, name='FIGSR')
