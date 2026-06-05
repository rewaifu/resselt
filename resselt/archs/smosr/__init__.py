from typing import Mapping

from .arch import SMoSR
from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import get_seq_len


class SMoSRArch(Architecture[SMoSR]):
    def __init__(self):
        super().__init__(
            uid='SMoSR',
            detect=KeyCondition.has_all(
                'short.weight',
                'short.bias',
                'blocks_1.0.short.weight',
                'blocks_1.0.short.bias',
                'blocks_1.0.body.0.eval_conv.weight',
                'blocks_1.0.body.0.eval_conv.bias',
                'blocks_1.0.body.2.eval_conv.weight',
                'blocks_1.0.body.2.eval_conv.bias',
                'blocks_1.0.body.4.eval_conv.weight',
                'blocks_1.0.body.4.eval_conv.bias',
                'blocks_1.1.body.0.eval_conv.weight',
                'blocks_1.1.body.0.eval_conv.bias',
                'blocks_1.1.body.2.eval_conv.weight',
                'blocks_1.1.body.2.eval_conv.bias',
                'blocks_1.1.body.4.eval_conv.weight',
                'blocks_1.1.body.4.eval_conv.bias',
                'blocks_2.0.body.0.eval_conv.weight',
                'blocks_2.0.body.0.eval_conv.bias',
                'blocks_2.0.body.2.eval_conv.weight',
                'blocks_2.0.body.2.eval_conv.bias',
                'blocks_2.0.body.4.eval_conv.weight',
                'blocks_2.0.body.4.eval_conv.bias',
                'end_block.0.body.0.eval_conv.weight',
                'end_block.0.body.0.eval_conv.bias',
                'end_block.0.body.2.eval_conv.weight',
                'end_block.0.body.2.eval_conv.bias',
                'end_block.0.body.4.eval_conv.weight',
                'end_block.0.body.4.eval_conv.bias',
                'end_block.1.eval_conv.weight',
                'end_block.1.eval_conv.bias',
                'upsampler.MetaUpsample',
            ),
        )

    def load(self, state_dict: Mapping[str, object]):
        state = state_dict

        upsamplers = ['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample', 'pa_up']
        dim, in_ch = state['blocks_1.0.body.0.eval_conv.weight'].shape[:2]
        n_mb = get_seq_len(state, 'blocks_2')
        _, upsampler, scale, _, out_dim, mid_dim, group, rep = [int(i) for i in state['upsampler.MetaUpsample']]
        d_conv = int(state['upsampler.2.end_conv.weight'].shape[2]) if upsampler == 4 else 1
    
        unpsampler = upsamplers[upsampler]
        # print(rep, dim, in_ch, n_mb, unpsampler, out_dim, mid_dim, group, scale, d_conv)
        model = SMoSR(
            in_ch=in_ch,
            out_ch=out_dim,
            dim=dim,
            scale=scale,
            rep=bool(rep),
            n_mb=n_mb,
            upsampler=unpsampler,
            upsampler_mid_dim=mid_dim,
            d_kernel=d_conv,
        )

        return self._enhance_model(model=model, in_channels=in_ch, out_channels=out_dim, upscale=scale, name='SMoSR')
