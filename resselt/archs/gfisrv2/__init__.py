from typing import Mapping
from .arch import GFISRV2
from ...factory import KeyCondition, Architecture
from ...utilities.state_dict import get_seq_len


class GFISRV2Arch(Architecture[GFISRV2]):
    def __init__(self):
        super().__init__(
            uid='GFISRV2',
            detect=KeyCondition.has_all(
                'gfisr_body.0.gamma',
                'gfisr_body.0.norm.scale',
                'gfisr_body.0.norm.offset',
                'gfisr_body.0.fc1.weight',
                'gfisr_body.0.fc1.bias',
                'gfisr_body.0.conv.pconv.rn.scale',
                'gfisr_body.0.conv.pconv.rn.offset',
                'gfisr_body.0.conv.pconv.post_norm.scale',
                'gfisr_body.0.conv.pconv.post_norm.offset',
                'gfisr_body.0.conv.pconv.fdc.weight',
                'gfisr_body.0.conv.pconv.fdc.bias',
                'gfisr_body.0.conv.pconv.fpe.weight',
                'gfisr_body.0.conv.pconv.fpe.bias',
                'gfisr_body.0.conv.dwconv_hw.weight',
                'gfisr_body.0.conv.dwconv_hw.bias',
                'gfisr_body.0.conv.dwconv_w.weight',
                'gfisr_body.0.conv.dwconv_w.bias',
                'gfisr_body.0.conv.dwconv_h.weight',
                'gfisr_body.0.conv.dwconv_h.bias',
                'gfisr_body.0.fc2.weight',
                'gfisr_body.0.fc2.bias',
                'upscale.MetaUpsample',
            ),
        )

    def load(self, state: Mapping[str, object]):
        samplemods = [
            "conv",
            "pixelshuffledirect",
            "pixelshuffle",
            "nearest+conv",
            "dysample",
            "transpose+conv",
            "lda",
            "pa_up",
        ]
        _, upsampler, scale, dim, out_ch, mid_dim, _ = [i.item() for i in state["upscale.MetaUpsample"]]
        upsampler = samplemods[upsampler]
        n_blocks = get_seq_len(state, "gfisr_body") - 3
        expansion_ratio = state["gfisr_body.0.fc1.weight"].shape[0] // 2 / dim
        if "in_to_dim.weight" in state:
            pixel_unshuffle = False
            in_nc = state["in_to_dim.weight"].shape[1]
        else:
            in_nc = state['in_to_dim.1.weight'].shape[1]
            if in_nc % 16 == 0:
                in_nc //= 16
            else:
                in_nc //= 4
            pixel_unshuffle = True
        model = GFISRV2(
            in_nc=in_nc,
            dim=dim,
            expansion_ratio=expansion_ratio,
            scale=scale,
            out_nc=out_ch,
            upsampler=upsampler,
            mid_dim=mid_dim,
            pixel_unshuffle=pixel_unshuffle,
            n_blocks=n_blocks,
        )

        return self._enhance_model(model=model, in_channels=int(in_nc), out_channels=int(out_ch), upscale=scale, name='GFISRV2')
