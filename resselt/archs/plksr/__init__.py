from typing import Mapping, Union

from .plksr import plksr
from .rplksr import realplksr
from resselt.utils import get_seq_len, pixelshuffle_scale
from resselt.registry.key_condition import KeyCondition
from resselt.registry.architecture import WrappedModel, Architecture

_PLKSR = Union[plksr, realplksr]


class PLKSRArch(Architecture[_PLKSR]):
    def __init__(self):
        super().__init__(
            id='PLKSR',
            detect=KeyCondition.has_all(
                'feats.0.weight',
                KeyCondition.has_any(
                    'feats.1.lk.conv.weight',
                    'feats.1.lk.convs.0.weight',
                    'feats.1.lk.mn_conv.weight',
                ),
                'feats.1.refine.weight',
                KeyCondition.has_any(
                    'feats.1.channe_mixer.0.weight',
                    'feats.1.channel_mixer.0.weight',
                ),
            ),
        )

    def load(self, state_dict: Mapping[str, object]) -> WrappedModel:
        dim = 64
        n_blocks = 28
        scale = 4
        kernel_size = 17
        split_ratio = 0.25
        use_ea = True
        in_nc = state_dict['feats.0.weight'].shape[1]
        out_nc = in_nc

        dim = state_dict['feats.0.weight'].shape[0]
        name = 'PLKSR'
        total_feat_layers = get_seq_len(state_dict, 'feats')
        use_ea = 'feats.1.attn.f.0.weight' in state_dict
        scale = pixelshuffle_scale(state_dict[f'feats.{total_feat_layers - 1}.weight'].shape[0], out_nc)
        if 'feats.1.channe_mixer.0.weight' in state_dict:
            # Yes, the normal version has this typo.
            n_blocks = total_feat_layers - 2

            # ccm_type
            mixer_0_shape = state_dict['feats.1.channe_mixer.0.weight'].shape[2]
            mixer_2_shape = state_dict['feats.1.channe_mixer.2.weight'].shape[2]
            if mixer_0_shape == 3 and mixer_2_shape == 1:
                ccm_type = 'CCM'
            elif mixer_0_shape == 3 and mixer_2_shape == 3:
                ccm_type = 'DCCM'
            elif mixer_0_shape == 1 and mixer_2_shape == 3:
                ccm_type = 'ICCM'
            else:
                raise ValueError('Unknown CCM type')

            # lk_type
            lk_type = 'PLK'
            use_max_kernel: bool = False
            sparse_kernels = [5, 5, 5, 5]
            sparse_dilations = [1, 2, 3, 4]
            with_idt: bool = False  # undetectable

            if 'feats.1.lk.conv.weight' in state_dict:
                # PLKConv2d
                lk_type = 'PLK'
                kernel_size = state_dict['feats.1.lk.conv.weight'].shape[2]
                split_ratio = state_dict['feats.1.lk.conv.weight'].shape[0] / dim
            elif 'feats.1.lk.convs.0.weight' in state_dict:
                # SparsePLKConv2d
                lk_type = 'SparsePLK'
                split_ratio = state_dict['feats.1.lk.convs.0.weight'].shape[0] / dim
                # Detecting other parameters for SparsePLKConv2d is praticaly impossible.
                # We cannot detect the values of sparse_dilations at all, we only know it has the same length as sparse_kernels.
                # Detecting the values of sparse_kernels is possible, but we don't know its length exactly, because it's `len(sparse_kernels) = len(convs) - (1 if use_max_kernel else 0)`.
                # However, we cannot detect use_max_kernel, because the convolutions it adds when enabled look the same as the other convolutions.
                # So I give up.
            elif 'feats.1.lk.mn_conv.weight' in state_dict:
                # RectSparsePLKConv2d
                lk_type = 'RectSparsePLK'
                kernel_size = state_dict['feats.1.lk.mn_conv.weight'].shape[2]
                split_ratio = state_dict['feats.1.lk.mn_conv.weight'].shape[0] / dim
            else:
                raise ValueError('Unknown LK type')

            model = plksr(
                dim=dim,
                n_blocks=n_blocks,
                upscaling_factor=scale,
                ccm_type=ccm_type,
                kernel_size=kernel_size,
                split_ratio=split_ratio,
                lk_type=lk_type,
                use_max_kernel=use_max_kernel,
                sparse_kernels=sparse_kernels,
                sparse_dilations=sparse_dilations,
                with_idt=with_idt,
                use_ea=use_ea,
            )
        elif 'feats.1.channel_mixer.0.weight' in state_dict:
            # and RealPLKSR doesn't. This makes it really convenient to detect.
            name = 'RealPLKSR'

            n_blocks = total_feat_layers - 3
            kernel_size = state_dict['feats.1.lk.conv.weight'].shape[2]
            split_ratio = state_dict['feats.1.lk.conv.weight'].shape[0] / dim
            dysample = False
            if 'to_img.init_pos' in state_dict:
                dysample = True
            model = realplksr(
                dim=dim,
                upscaling_factor=scale,
                n_blocks=n_blocks,
                kernel_size=kernel_size,
                split_ratio=split_ratio,
                use_ea=use_ea,
                norm_groups=4,  # un-detectable
                dysample=dysample,
            )
        else:
            raise ValueError('Unknown model type')
        return WrappedModel(model=model, in_channels=in_nc, out_channels=out_nc, upscale=scale, name=name)
