import math
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
import numpy as np

SampleMods = Literal['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample', 'transpose+conv', 'lda', 'pa_up']


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel=1,
    ) -> None:
        super().__init__()

        if in_channels <= groups or in_channels % groups != 0:
            msg = 'Incorrect in_channels and groups values.'
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, end_kernel, 1, end_kernel // 2)
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij'))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device, pin_memory=True).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode='bilinear',
            align_corners=False,
            padding_mode='border',
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


class LayerNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x):
        if x.is_contiguous(memory_format=torch.channels_last):
            return F.layer_norm(x.permute(0, 2, 3, 1), self.dim, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LDA_AQU(nn.Module):
    def __init__(
        self,
        in_channels=48,
        reduction_factor=4,
        nh=1,
        scale_factor=2.0,
        k_e=3,
        k_u=3,
        n_groups=2,
        range_factor=11,
        rpb=True,
    ):
        super(LDA_AQU, self).__init__()
        self.k_u = k_u
        self.num_head = nh
        self.scale_factor = scale_factor
        self.n_groups = n_groups
        self.offset_range_factor = range_factor

        self.attn_dim = in_channels // (reduction_factor * self.num_head)
        self.scale = self.attn_dim**-0.5
        self.rpb = rpb
        self.hidden_dim = in_channels // reduction_factor
        self.proj_q = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.proj_k = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.group_channel = in_channels // (reduction_factor * self.n_groups)
        # print(self.group_channel)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.group_channel,
                self.group_channel,
                3,
                1,
                1,
                groups=self.group_channel,
                bias=False,
            ),
            LayerNorm(self.group_channel),
            nn.SiLU(),
            nn.Conv2d(self.group_channel, 2 * k_u**2, k_e, 1, k_e // 2),
        )
        print(2 * k_u**2)
        self.layer_norm = LayerNorm(in_channels)

        self.pad = int((self.k_u - 1) / 2)
        base = np.arange(-self.pad, self.pad + 1).astype(np.float32)
        base_y = np.repeat(base, self.k_u)
        base_x = np.tile(base, self.k_u)
        base_offset = np.stack([base_y, base_x], axis=1).flatten()
        base_offset = torch.tensor(base_offset).view(1, -1, 1, 1)
        self.register_buffer('base_offset', base_offset, persistent=False)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(torch.zeros(1, self.num_head, 1, self.k_u**2, self.hidden_dim // self.num_head))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def get_offset(self, offset, Hout, Wout):
        B, _, _, _ = offset.shape
        device = offset.device
        row_indices = torch.arange(Hout, device=device)
        col_indices = torch.arange(Wout, device=device)
        row_indices, col_indices = torch.meshgrid(row_indices, col_indices)
        index_tensor = torch.stack((row_indices, col_indices), dim=-1).view(1, Hout, Wout, 2)
        offset = rearrange(offset, 'b (kh kw d) h w -> b kh h kw w d', kh=self.k_u, kw=self.k_u)
        offset = offset + index_tensor.view(1, 1, Hout, 1, Wout, 2)
        offset = offset.contiguous().view(B, self.k_u * Hout, self.k_u * Wout, 2)

        offset[..., 0] = 2 * offset[..., 0] / (Hout - 1) - 1
        offset[..., 1] = 2 * offset[..., 1] / (Wout - 1) - 1
        offset = offset.flip(-1)
        return offset

    def extract_feats(self, x, offset, ks=3):
        out = nn.functional.grid_sample(x, offset, mode='bilinear', padding_mode='zeros', align_corners=True)
        out = rearrange(out, 'b c (ksh h) (ksw w) -> b (ksh ksw) c h w', ksh=ks, ksw=ks)
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        out_H, out_W = int(H * self.scale_factor), int(W * self.scale_factor)
        v = x
        x = self.layer_norm(x)
        q = self.proj_q(x)
        k = self.proj_k(x)

        q = torch.nn.functional.interpolate(q, (out_H, out_W), mode='bilinear', align_corners=True)
        q_off = q.view(B * self.n_groups, -1, out_H, out_W)
        pred_offset = self.conv_offset(q_off)
        offset = pred_offset.tanh().mul(self.offset_range_factor) + self.base_offset.to(x.dtype)

        k = k.view(B * self.n_groups, self.hidden_dim // self.n_groups, H, W)
        v = v.view(B * self.n_groups, C // self.n_groups, H, W)
        offset = self.get_offset(offset, out_H, out_W)
        k = self.extract_feats(k, offset=offset)
        v = self.extract_feats(v, offset=offset)

        q = rearrange(q, 'b (nh c) h w -> b nh (h w) () c', nh=self.num_head)
        k = rearrange(k, '(b g) n c h w -> b (h w) n (g c)', g=self.n_groups)
        v = rearrange(v, '(b g) n c h w -> b (h w) n (g c)', g=self.n_groups)
        k = rearrange(k, 'b n1 n (nh c) -> b nh n1 n c', nh=self.num_head)
        v = rearrange(v, 'b n1 n (nh c) -> b nh n1 n c', nh=self.num_head)

        if self.rpb:
            k = k + self.relative_position_bias_table

        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, 'b nh (h w) t c -> b (nh c) (t h) w', h=out_H)
        return out


class PA(nn.Module):
    def __init__(self, dim):
        super(PA, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x):
        return x.mul(self.conv(x))


class UniUpsampleV3(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods = 'pa_up',
        scale: int = 2,
        in_dim: int = 48,
        out_dim: int = 3,
        mid_dim: int = 48,
        group: int = 4,  # Only DySample
        dysample_end_kernel=1,  # needed only for compatibility with version 2
    ) -> None:
        m = []

        if scale == 1 or upsample == 'conv':
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == 'pixelshuffledirect':
            m.extend([nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)])
        elif upsample == 'pixelshuffle':
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend([nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)])
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == 'nearest+conv':
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == 'dysample':
            if mid_dim != in_dim:
                m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            m.append(DySample(mid_dim, out_dim, scale, group, end_kernel=dysample_end_kernel))
            # m.append(nn.Conv2d(mid_dim, out_dim, dysample_end_kernel, 1, dysample_end_kernel//2)) # kernel 1 causes chromatic artifacts
        elif upsample == 'transpose+conv':
            if scale == 2:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1))
            elif scale == 3:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 3, 3, 0))
            elif scale == 4:
                m.extend(
                    [
                        nn.ConvTranspose2d(in_dim, in_dim, 4, 2, 1),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                    ]
                )
            else:
                raise ValueError(f'scale {scale} is not supported. Supported scales: 2, 3, 4')
            m.append(nn.Conv2d(out_dim, out_dim, 3, 1, 1))
        elif upsample == 'lda':
            if mid_dim != in_dim:
                m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            m.append(LDA_AQU(mid_dim, scale_factor=scale))
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == 'pa_up':
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                            PA(mid_dim),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ]
                    )
                    in_dim = mid_dim
            elif scale == 3:
                m.extend(
                    [
                        nn.Upsample(scale_factor=3),
                        nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                        PA(mid_dim),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
            else:
                raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        else:
            raise ValueError(f'An invalid Upsample was selected. Please choose one of {SampleMods}')
        super().__init__(*m)

        self.register_buffer(
            'MetaUpsample',
            torch.tensor(
                [
                    3,  # Block version, if you change something, please number from the end so that you can distinguish between authorized changes and third parties
                    list(SampleMods.__args__).index(upsample),  # UpSample method index
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, s=1, bias=True) -> None:
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
        if self.training is False:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False
            self.update_params()

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)

        sk_w = self.sk.weight.data.clone().detach()

        if self.bias:
            b1 = self.conv[0].bias.data.clone().detach()
            b2 = self.conv[1].bias.data.clone().detach()
            b3 = self.conv[2].bias.data.clone().detach()
            b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
            self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
            sk_b = self.sk.bias.data.clone().detach()

        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])
        self.weight_concat = self.weight_concat + sk_w
        self.eval_conv.weight.data = self.weight_concat
        if self.bias:
            self.bias_concat = self.bias_concat + sk_b
            self.eval_conv.bias.data = self.bias_concat

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.update_params()
        return self

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), 'constant', 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            out = self.eval_conv(x)
        return out


class SPAB(nn.Module):
    def __init__(self, in_channels, mid_dim=None, out_dim=None, bias=False, end=False) -> None:
        super().__init__()
        mid_dim = mid_dim if mid_dim else in_channels
        out_dim = out_dim if out_dim else in_channels
        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_dim, gain1=2, s=1, bias=bias)
        self.c2_r = Conv3XC(mid_dim, mid_dim, gain1=2, s=1, bias=bias)
        self.c3_r = Conv3XC(mid_dim, out_dim, gain1=2, s=1, bias=bias)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.end = end

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        out3 = self.c3_r(out2_act)
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        if self.end:
            return out, out1
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.offset = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-1.0 / 2))
        x_normed = x / (rms_x + self.eps)
        return self.scale[..., None, None] * x_normed + self.offset[..., None, None]


class InceptionDWConv2d(nn.Module):
    """Inception depthweise convolution"""

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125) -> None:
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class Attention(nn.Module):
    def __init__(self, dim, num_heads) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, 1, dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Преобразуем в (b, num_heads, head_dim, h*w)
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        q = F.normalize(q, dim=3)
        k = F.normalize(k, dim=3)

        attn = torch.matmul(q, k.transpose(2, 3)) * self.temperature  # (b, num_heads, hw, hw)
        attn = attn.softmax(dim=3)

        out = torch.matmul(attn, v)  # (b, num_heads, head_dim, hw)

        # Обратно в (b, c, h, w)
        out = out.view(b, c, h, w)

        out = self.project_out(out)
        return out


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 1.5,
        conv_ratio: float = 1,
        att=False,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1, 1)

        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.token_mix = Attention(conv_channels, 16) if att else InceptionDWConv2d(dim)
        self.fc2 = nn.Conv2d(hidden, dim, 1, 1, 0)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.token_mix(c)
        x = self.act(g) * torch.cat((i, c), dim=1)
        x = self.act(self.fc2(x))
        return x


class SimpleGate(nn.Module):
    @staticmethod
    def forward(x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MetaGated(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        hidden = dim * 2
        self.local = nn.Sequential(
            RMSNorm(dim),
            nn.Conv2d(dim, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=dim),
            SimpleGate(),
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
        )
        self.glob = GatedCNNBlock(dim)
        self.gamma0 = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.gamma1 = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        short = x
        x = self.local(x)
        x = x * self.sca(x)
        x = x * self.gamma0 + short
        del short
        x = self.glob(x) * self.gamma1 + x
        return x


class Down(nn.Sequential):
    def __init__(self, dim) -> None:
        super().__init__(nn.Conv2d(dim, dim // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))


class Upsample(nn.Sequential):
    def __init__(self, dim) -> None:
        super().__init__(nn.Conv2d(dim, dim * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))


class Block(nn.Module):
    def __init__(self, dim, num_gated, down=True) -> None:
        super().__init__()

        if down:
            self.gated = nn.Sequential(*[MetaGated(dim) for _ in range(num_gated)])
            self.scale = Down(dim)

        else:
            self.scale = Upsample(dim)
            self.gated = nn.Sequential(*[MetaGated(dim // 2) for _ in range(num_gated)])
            self.shor = nn.Conv2d(int(dim), dim // 2, 1, 1, 0)
        self.down = down

    def forward(self, x, short=None):
        if self.down:
            x = self.gated(x)
            return self.scale(x), x
        else:
            x = torch.cat([self.scale(x), short], dim=1)
            x = self.shor(x)
            return self.gated(x)


class GateRV3(nn.Module):
    def __init__(
        self,
        in_ch=3,
        dim=32,
        enc_blocks=(2, 2, 4, 8),
        dec_blocks=(2, 2, 2, 2),
        num_latent=12,
        scale=1,
        upsample: SampleMods = 'pixelshuffle',
        upsample_mid_dim=32,
        end_gamma_init=1,
        attention=False,
        span_blocks=4,
        end_kernel=3,
        **kwargs,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)
        self.gater_encode = nn.ModuleList([Block(dim * (2**i), enc_blocks[i]) for i in range(len(enc_blocks))])
        self.span_block0 = SPAB(dim, end=False)
        self.span_n_b = nn.Sequential(*[SPAB(dim, end=False) for _ in range(span_blocks)])
        self.span_end = SPAB(dim, end=True)
        self.sisr_end_conv = Conv3XC(dim, dim, bias=True)
        self.sisr_cat_conv = nn.Conv2d(dim * 4, dim, 1)
        nn.init.trunc_normal_(self.sisr_cat_conv.weight, std=0.02)
        self.latent = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim * (2 ** len(enc_blocks)),
                    expansion_ratio=1.5,
                    conv_ratio=1.00,
                    att=attention,
                )
                for _ in range(num_latent)
            ]
        )
        self.decode = nn.ModuleList(
            [
                Block(
                    dim * (2 ** (len(dec_blocks) - i)),
                    dec_blocks[i],
                    False,
                )
                for i in range(len(dec_blocks))
            ]
        )
        self.pad = 2 ** (len(enc_blocks))
        # self.dim_to_in = nn.Conv2d(dim*2, in_ch, 3, 1, 1)
        self.gamma = nn.Parameter(torch.ones(1, in_ch, 1, 1) * end_gamma_init)
        self.gamma.register_hook(lambda grad: grad * 100)
        if scale != 1:
            self.short_to_dim = nn.Upsample(scale_factor=scale)  # ConvBlock(in_ch, dim)
            self.dim_to_in = UniUpsampleV3(upsample, scale, dim, in_ch, upsample_mid_dim, dysample_end_kernel=end_kernel)
            # self.upsample =
        else:
            self.dim_to_in = nn.Conv2d(dim, in_ch, 3, 1, 1)
            self.short_to_dim = nn.Identity()

    def load_state_dict(self, state_dict, *args, **kwargs):
        if 'dim_to_in.MetaUpsample' in state_dict:
            state_dict['dim_to_in.MetaUpsample'] = self.dim_to_in.MetaUpsample
        if 'gamma' not in state_dict:
            state_dict['gamma'] = self.gamma
        return super().load_state_dict(state_dict, *args, **kwargs)

    def check_img_size(self, x, resolution: tuple[int, int]):
        scaled_size = self.pad
        mod_pad_h = (scaled_size - resolution[0] % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - resolution[1] % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_img_size(inp, (H, W))
        x = self.in_to_dim(inp)
        sisr = self.span_block0(x)
        sisr_short = sisr
        sisr = self.span_n_b(sisr)
        sisr, sisr_out = self.span_end(sisr)
        sisr = self.sisr_end_conv(sisr)
        sisr = self.sisr_cat_conv(torch.cat([x, sisr, sisr_short, sisr_out], dim=1))
        del sisr_short, sisr_out
        shorts = []
        for block in self.gater_encode:
            x, short = block(x)
            shorts.append(short)

        x = self.latent(x)
        len_block = len(self.decode)
        shorts.reverse()
        for index in range(len_block):
            x = self.decode[index](x, shorts[index])

        x = self.dim_to_in(x + sisr) + self.gamma * self.short_to_dim(inp)
        return x[:, :, : H * self.scale, : W * self.scale]
