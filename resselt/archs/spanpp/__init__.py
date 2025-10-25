from typing import Mapping

from .arch import SpanPP
from ...factory import Architecture, KeyCondition
from ...utilities.state_dict import get_seq_len


class SpanPPArch(Architecture[SpanPP]):
    def __init__(self):
        super().__init__(
            uid='SpanPP',
            detect=KeyCondition.has_all(
                'conv0.alpha',
                'conv0.conv1.k0',
                'conv0.conv1.b0',
                'conv0.conv1.k1',
                'conv0.conv1.b1',
                'conv0.conv2.weight',
                'conv0.conv2.bias',
                'conv0.conv3.sk.weight',
                'conv0.conv3.sk.bias',
                'conv0.conv3.conv.0.weight',
                'conv0.conv3.conv.0.bias',
                'conv0.conv3.conv.1.weight',
                'conv0.conv3.conv.1.bias',
                'conv0.conv3.conv.2.weight',
                'conv0.conv3.conv.2.bias',
                'conv0.conv3.eval_conv.weight',
                'conv0.conv3.eval_conv.bias',
                'conv0.conv_3x3_rep.weight',
                'conv0.conv_3x3_rep.bias',
                'block_1.c1_r.alpha',
                'block_1.c1_r.conv1.k0',
                'block_1.c1_r.conv1.b0',
                'block_1.c1_r.conv1.k1',
                'block_1.c1_r.conv1.b1',
                'block_1.c1_r.conv2.weight',
                'block_1.c1_r.conv2.bias',
                'block_1.c1_r.conv3.sk.weight',
                'block_1.c1_r.conv3.sk.bias',
                'block_1.c1_r.conv3.conv.0.weight',
                'block_1.c1_r.conv3.conv.0.bias',
                'block_1.c1_r.conv3.conv.1.weight',
                'block_1.c1_r.conv3.conv.1.bias',
                'block_1.c1_r.conv3.conv.2.weight',
                'block_1.c1_r.conv3.conv.2.bias',
                'block_1.c1_r.conv3.eval_conv.weight',
                'block_1.c1_r.conv3.eval_conv.bias',
                'block_1.c1_r.conv_3x3_rep.weight',
                'block_1.c1_r.conv_3x3_rep.bias',
                'block_1.c2_r.alpha',
                'block_1.c2_r.conv1.k0',
                'block_1.c2_r.conv1.b0',
                'block_1.c2_r.conv1.k1',
                'block_1.c2_r.conv1.b1',
                'block_1.c2_r.conv2.weight',
                'block_1.c2_r.conv2.bias',
                'block_1.c2_r.conv3.sk.weight',
                'block_1.c2_r.conv3.sk.bias',
                'block_1.c2_r.conv3.conv.0.weight',
                'block_1.c2_r.conv3.conv.0.bias',
                'block_1.c2_r.conv3.conv.1.weight',
                'block_1.c2_r.conv3.conv.1.bias',
                'block_1.c2_r.conv3.conv.2.weight',
                'block_1.c2_r.conv3.conv.2.bias',
                'block_1.c2_r.conv3.eval_conv.weight',
                'block_1.c2_r.conv3.eval_conv.bias',
                'block_1.c2_r.conv_3x3_rep.weight',
                'block_1.c2_r.conv_3x3_rep.bias',
                'block_1.c3_r.alpha',
                'block_1.c3_r.conv1.k0',
                'block_1.c3_r.conv1.b0',
                'block_1.c3_r.conv1.k1',
                'block_1.c3_r.conv1.b1',
                'block_1.c3_r.conv2.weight',
                'block_1.c3_r.conv2.bias',
                'block_1.c3_r.conv3.sk.weight',
                'block_1.c3_r.conv3.sk.bias',
                'block_1.c3_r.conv3.conv.0.weight',
                'block_1.c3_r.conv3.conv.0.bias',
                'block_1.c3_r.conv3.conv.1.weight',
                'block_1.c3_r.conv3.conv.1.bias',
                'block_1.c3_r.conv3.conv.2.weight',
                'block_1.c3_r.conv3.conv.2.bias',
                'block_1.c3_r.conv3.eval_conv.weight',
                'block_1.c3_r.conv3.eval_conv.bias',
                'block_1.c3_r.conv_3x3_rep.weight',
                'block_1.c3_r.conv_3x3_rep.bias',
                'block_2.c1_r.alpha',
                'block_2.c1_r.conv1.k0',
                'block_2.c1_r.conv1.b0',
                'block_2.c1_r.conv1.k1',
                'block_2.c1_r.conv1.b1',
                'block_2.c1_r.conv2.weight',
                'block_2.c1_r.conv2.bias',
                'block_2.c1_r.conv3.sk.weight',
                'block_2.c1_r.conv3.sk.bias',
                'block_2.c1_r.conv3.conv.0.weight',
                'block_2.c1_r.conv3.conv.0.bias',
            ),
        )

    def load(self, state_dict: Mapping[str, object]):
        state = state_dict
        dim, in_ch = state['conv0.conv_3x3_rep.weight'].shape[:2]
        if "MetaIGConv" in state:
            scales = state["MetaIGConv"].tolist()
        else:
            scales = [1, 2, 3, 4]
        ig_kernel, implicit_dim = state["upsampler.freq"].shape[:2]
        ig_kernel_size = int((ig_kernel / dim) ** 0.5)
        latent_layers = get_seq_len(state, "upsampler.query_kernel") // 2
        model = SpanPP(
            num_in_ch=in_ch,
            feature_channels=dim,
            scale_list=scales,
            eval_base_scale=2,
            ig_kernel=ig_kernel_size,
            implicit_dim=implicit_dim,
            latent_layers=latent_layers,
        )

        return self._enhance_model(model=model, in_channels=in_ch, out_channels=in_ch, upscale=scales, name='SpanPP')
