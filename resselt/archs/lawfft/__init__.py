from typing import Mapping

from .arch import LAWFFT
from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import get_seq_len


class LAWFFTArch(Architecture[LAWFFT]):
    def __init__(self):
        super().__init__(
            uid='LAWFFT',
            detect=KeyCondition.has_all(
                'in_to_dim.weight',
                'in_to_dim.bias',
                'body.0.residual.0.token_mix.0.weight',
                'body.0.residual.0.token_mix.0.bias',
                'body.0.residual.0.token_mix.1.local.0.kernel_gen.1.weight',
                'body.0.residual.0.token_mix.1.local.0.kernel_gen.1.bias',
                'body.0.residual.0.token_mix.1.local.0.kernel_gen.3.weight',
                'body.0.residual.0.token_mix.1.local.0.kernel_gen.3.bias',
                'body.0.residual.0.token_mix.1.local.1.kernel_gen.1.weight',
                'body.0.residual.0.token_mix.1.local.1.kernel_gen.1.bias',
                'body.0.residual.0.token_mix.1.local.1.kernel_gen.3.weight',
                'body.0.residual.0.token_mix.1.local.1.kernel_gen.3.bias',
                'body.0.residual.0.token_mix.1.att.to_hidden.weight',
                'body.0.residual.0.token_mix.1.att.to_hidden.bias',
                'body.0.residual.0.token_mix.1.att.to_hidden_dw.weight',
                'body.0.residual.0.token_mix.1.att.to_hidden_dw.bias',
                'body.0.residual.0.token_mix.1.att.project_out.weight',
                'body.0.residual.0.token_mix.1.att.project_out.bias',
                'body.0.residual.0.token_mix.1.att.norm.weight',
                'body.0.residual.0.token_mix.1.att.norm.bias',
                'body.0.residual.0.token_mix.1.last.weight',
                'body.0.residual.0.token_mix.1.last.bias',
                'body.0.residual.0.channel_mix1.0.weight',
                'body.0.residual.0.channel_mix1.0.bias',
                'body.0.residual.0.channel_mix1.1.project_in.weight',
                'body.0.residual.0.channel_mix1.1.project_in.bias',
                'body.0.residual.0.channel_mix1.1.dwconv.weight',
                'body.0.residual.0.channel_mix1.1.dwconv.bias',
                'body.0.residual.0.channel_mix1.1.project_out.weight',
                'body.0.residual.0.channel_mix1.1.project_out.bias',
            ),
        )

    def load(self, state_dict: Mapping[str, object]):
        samplemods = ['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample']
        _, upsampler, scale, dim, in_ch, mid_dim, _ = [i.item() for i in state_dict['upscale.MetaUpsample']]
        upsampler = samplemods[upsampler]
        unshuffle_mod = 'in_to_dim.1.weight' in state_dict
        window_size = state_dict['window_size'].item()
        local_dim = state_dict['body.0.residual.0.token_mix.1.local.0.kernel_gen.1.bias'].shape[0]
        split = 1 / (dim / local_dim)
        n_rblock = get_seq_len(state_dict, 'body')
        n_mblock = get_seq_len(state_dict, 'body.0.residual') - 1
        global_dim = dim - int((dim * split))
        t_mid_factor = state_dict['body.0.residual.1.token_mix.1.att.to_hidden.bias'].shape[0] / global_dim / 3
        mlp_factor = state_dict['body.0.residual.1.channel_mix1.1.project_in.bias'].shape[0] / dim / 2
        model = LAWFFT(
            in_ch=in_ch,
            dim=dim,
            split=split,
            scale=scale,
            n_rblock=n_rblock,
            n_mblock=n_mblock,
            t_mid_factor=t_mid_factor,
            window_size=window_size,
            mlp_factor=mlp_factor,
            unshuffle_mod=unshuffle_mod,
            upsampler=upsampler,
            mid_dim=mid_dim,
        )

        return self._enhance_model(model=model, in_channels=in_ch, out_channels=in_ch, upscale=scale, name='LAWFFT')
