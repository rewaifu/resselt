#https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel/spandrel/architectures/FDAT/__init__.py
import math
from typing import Mapping

from .arch import FDAT, SampleMods3
from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import get_seq_len


class FDATArch(Architecture[FDAT]):
    def __init__(self):
        super().__init__(
            uid='FDAT',
            detect=KeyCondition.has_all(
                "groups.0.blocks.0.attn.bias",
                "groups.0.blocks.0.inter.cg.1.weight",
                "groups.0.blocks.0.ffn.fc1.weight",
                "groups.0.blocks.0.n1.weight",
                "upsampler.MetaUpsample",
            ),
        )

    def load(self, state_dict: Mapping[str, object]):
        state = state_dict
        _, upsampler_index, scale, embed_dim, num_out_ch, mid_dim, _ = state_dict[
                    "upsampler.MetaUpsample"
                ].tolist()
        upsampler_type = list(SampleMods3.__args__)[upsampler_index]

        if "conv_first.1.weight" in state_dict:
            num_in_ch = num_out_ch
            scale = 4 // (
                math.isqrt(state_dict["conv_first.1.weight"].shape[1] // num_in_ch)
            )
            unshuffle_mod = True
        else:
            unshuffle_mod = False
            num_in_ch = state_dict["conv_first.weight"].shape[1]

        num_groups = get_seq_len(state_dict, "groups")
        group_block_pattern = ["spatial", "channel"]
        depth_per_group = get_seq_len(state_dict, "groups.0.blocks") // len(
            group_block_pattern
        )
        num_heads = state_dict["groups.0.blocks.0.attn.bias"].shape[0]
        window_size = math.isqrt(state_dict["groups.0.blocks.0.attn.bias"].shape[2])
        ffn_expansion_ratio = float(
            state_dict["groups.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim
        )
        aim_reduction_ratio = (
            embed_dim // state_dict["groups.0.blocks.0.inter.cg.1.weight"].shape[0]
        )

        img_range = 1.0

        model = FDAT(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            embed_dim=embed_dim,
            num_groups=num_groups,
            depth_per_group=depth_per_group,
            num_heads=num_heads,
            window_size=window_size,
            ffn_expansion_ratio=ffn_expansion_ratio,
            aim_reduction_ratio=aim_reduction_ratio,
            group_block_pattern=None,
            upsampler_type=upsampler_type,
            mid_dim=mid_dim,
            img_range=img_range,
            unshuffle_mod=unshuffle_mod,
        )

        return self._enhance_model(model=model, in_channels=num_in_ch, out_channels=num_out_ch, upscale=scale, name='FDAT')
