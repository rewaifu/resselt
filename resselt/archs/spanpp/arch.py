# type: ignore
from typing import Self
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import numpy as np


class Conv3XC(nn.Module):
    def __init__(self, c_in: int, c_out: int, gain1: int = 2, s: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.bias = bias
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        nn.init.trunc_normal_(self.sk.weight, std=0.02)

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w2 = self.conv[1].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w3 = self.conv[2].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)

        sk_w = self.sk.weight.data.clone().detach()

        if self.bias:
            b1 = self.conv[0].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b2 = self.conv[1].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b3 = self.conv[2].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
            self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
            sk_b = self.sk.bias.data.clone().detach()  # pyright: ignore[reportOptionalMemberAccess]

        target_kernel_size = 3

        h_pixels_to_pad = (target_kernel_size - 1) // 2
        w_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [h_pixels_to_pad, h_pixels_to_pad, w_pixels_to_pad, w_pixels_to_pad])
        self.weight_concat = self.weight_concat + sk_w
        self.eval_conv.weight.data = self.weight_concat
        if self.bias:
            self.bias_concat = self.bias_concat + sk_b  # pyright: ignore[reportOperatorIssue,reportPossiblyUnboundVariable]
            self.eval_conv.bias.data = self.bias_concat  # pyright: ignore[reportOptionalMemberAccess]

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode:
            self.update_params()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), 'constant', 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            out = self.eval_conv(x)
        return out


class SeqConv3x3(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier) -> None:
        super().__init__()
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.mid_planes = int(out_planes * depth_multiplier)
        conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
        self.k0 = conv0.weight
        self.b0 = conv0.bias

        conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
        self.k1 = conv1.weight
        self.b1 = conv1.bias

    def forward(self, x):
        # conv-1x1
        y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
        # explicitly padding with bias
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        # conv-3x3
        return F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None
        # re-param conv kernel
        RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
        # re-param conv bias
        RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
        RB = (
            F.conv2d(input=RB, weight=self.k1).view(
                -1,
            )
            + self.b1
        )
        return RK, RB


class RepConv(nn.Module):
    def __init__(self, in_dim=3, out_dim=32) -> None:
        super().__init__()
        self.conv1 = SeqConv3x3(in_dim, out_dim, 2)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv3 = Conv3XC(in_dim, out_dim)
        self.conv_3x3_rep = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.alpha = nn.Parameter(torch.randn(3), requires_grad=True)
        # self.alpha.register_hook(lambda grad: grad * 100)
        self.forward_module = self.train_forward

        nn.init.constant_(self.alpha, 1.0)

    def fuse(self) -> None:
        conv1_w, conv1_b = self.conv1.rep_params()
        conv2_w, conv2_b = self.conv2.weight, self.conv2.bias
        self.conv3.update_params()
        conv3_w, conv3_b = self.conv3.eval_conv.weight, self.conv3.eval_conv.bias
        device = self.conv_3x3_rep.weight.device
        sum_weight = (self.alpha[0] * conv1_w + self.alpha[1] * conv2_w + self.alpha[2] * conv3_w).to(device)
        sum_bias = (self.alpha[0] * conv1_b + self.alpha[1] * conv2_b + self.alpha[2] * conv3_b).to(device)
        self.conv_3x3_rep.weight = nn.Parameter(sum_weight)
        self.conv_3x3_rep.bias = nn.Parameter(sum_bias)

    def train_forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.alpha[0] * x1 + self.alpha[1] * x2 + self.alpha[2] * x3

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.fuse()
        return self

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.conv_3x3_rep(x)


class SPAB(nn.Module):
    def __init__(self, in_channels):
        super(SPAB, self).__init__()
        self.in_channels = in_channels
        self.c1_r = RepConv(in_channels, in_channels)
        self.c2_r = RepConv(in_channels, in_channels)
        self.c3_r = RepConv(in_channels, in_channels)
        self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1


def make_coord(shape, ranges=None):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = ret.flip(-1)
    return ret


class EvalConv(nn.Module):
    def __init__(self, weight, kernel_size):
        super(EvalConv, self).__init__()
        self.weight = weight
        self.kernel_size = kernel_size

    def forward(self, x):
        return F.conv2d(x, self.weight.to(x), bias=None, stride=1, padding=self.kernel_size // 2)


class IGConv(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size,
        implicit_dim: int = 256,
        latent_layers: int = 4,
        scale_list=(1, 2, 3, 4),
        base_scale=2,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        assert implicit_dim % 2 == 0
        self.implicit_dim = implicit_dim
        self.latent_layers = latent_layers
        self.base_scale = base_scale
        self.max_s = np.max(scale_list)

        self.phase = nn.Conv2d(1, implicit_dim // 2, 1, 1)
        self.freq = nn.Parameter(torch.randn((dim * kernel_size**2), implicit_dim, 1, 1) * 0.02, requires_grad=True)
        self.amplitude = nn.Parameter(torch.randn((dim * kernel_size**2), implicit_dim, 1, 1) * 0.02, requires_grad=True)
        query_kernel_layers = []
        for _ in range(latent_layers):
            query_kernel_layers.append(nn.Conv2d(implicit_dim, implicit_dim, 1, 1, 0))
            query_kernel_layers.append(nn.ReLU())

        query_kernel_layers.append(nn.Conv2d(implicit_dim, 3, 1, 1, 0))
        self.query_kernel = nn.Sequential(*query_kernel_layers)
        self.resize = self._implicit_representation_latent
        self.eval_convs = {}
        self.scale_list = scale_list

    def train(self, mode: bool = True) -> Self:
        with torch.no_grad():
            for scale in self.scale_list:
                self.eval_convs[str(scale)] = EvalConv(self.resize(scale), self.kernel_size)
        return super().train(mode)

    def forward(self, x, scale):
        scale = self.base_scale if scale is None else scale
        # if self.training:
        #     k_interp = self.resize(scale)
        #     rgb = F.conv2d(x, k_interp, bias=None, stride=1, padding=self.kernel_size // 2)
        # else:
        rgb = self.eval_convs[str(scale)](x)
        rgb = F.pixel_shuffle(rgb, scale)
        return rgb

    def _implicit_representation_latent(self, scale):
        scale_phase = min(scale, self.max_s)
        r = torch.ones(1, 1, scale, scale).to(self.query_kernel[0].weight.device) / scale_phase * 2  # 2 / r following LIIF/LTE
        coords = make_coord((scale, scale)).unsqueeze(0).to(self.query_kernel[0].weight.device)
        freq = self.freq.repeat(1, 1, scale, scale)  # RGB RGB
        amplitude = self.amplitude.repeat(1, 1, scale, scale)
        coords = coords.permute(0, 3, 1, 2).contiguous()

        # Fourier basis
        coords = coords.repeat(freq.shape[0], 1, 1, 1)
        freq_1, freq_2 = freq.chunk(2, dim=1)
        freq = freq_1 * coords[:, :1] + freq_2 * coords[:, 1:]  # RGB
        phase = self.phase(r)  # To RGB
        freq = freq + phase  # RGB
        freq = torch.cat([torch.cos(torch.pi * freq), torch.sin(torch.pi * freq)], dim=1)  # cos(R)cos(G)cos(B) sin(R)sin(G)sin(B)

        # 4. R(F_theta(.))
        k_interp = self.query_kernel(freq * amplitude)
        k_interp = rearrange(k_interp, '(Cin Kh Kw) RGB rh rw -> (RGB rh rw) Cin Kh Kw', Kh=self.kernel_size, Kw=self.kernel_size, Cin=self.dim)
        return k_interp


class SpanPP(nn.Module):
    """Swift Parameter-free Attention Network for Efficient Super-Resolution"""

    def __init__(
        self,
        num_in_ch=3,
        feature_channels=48,
        scale_list=(1, 2, 3, 4),
        eval_base_scale=2,
        ig_kernel_size=3,
        implicit_dim=256,
        latent_layers=4,
        **kwargs,
    ):
        super(SpanPP, self).__init__()
        scale_list = list(set(scale_list))
        in_channels = num_in_ch
        self.conv0 = RepConv(in_channels, feature_channels)
        self.block_1 = SPAB(feature_channels)
        self.block_2 = SPAB(feature_channels)
        self.block_3 = SPAB(feature_channels)
        self.block_4 = SPAB(feature_channels)
        self.block_5 = SPAB(feature_channels)
        self.block_6 = SPAB(feature_channels)

        self.conv_cat = nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = RepConv(feature_channels, feature_channels)
        self.scale_list = scale_list
        self.upsampler = IGConv(
            feature_channels, ig_kernel_size, implicit_dim, latent_layers, scale_list, eval_base_scale
        )  # nn.Sequential(nn.Conv2d(feature_channels,out_channels*scale*scale,3,1,1),nn.PixelShuffle(scale))
        self.register_buffer(
            'MetaIGConv',
            torch.tensor(
                list(scale_list),
                dtype=torch.uint8,
            ),
        )

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict['MetaIGConv'] = self.MetaIGConv
        return super().load_state_dict(state_dict, *args, **kwargs)

    def forward(self, x, scale=None):
        out_feature = self.conv0(x)

        out_b1, _ = self.block_1(out_feature)
        out_b2, _ = self.block_2(out_b1)
        out_b3, _ = self.block_3(out_b2)

        out_b4, _ = self.block_4(out_b3)
        out_b5, _ = self.block_5(out_b4)
        out_b6, out_b5_2 = self.block_6(out_b5)
        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        # if scale is None and self.training:
        #     scale = random.choice(self.scale_list)
        output = self.upsampler(out, scale)
        return output
