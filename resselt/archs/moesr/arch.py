import math
from typing import Literal

import torch
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn import functional as F

from resselt.archs.utils import DySample


class SSELayer(nn.Module):
    def __init__(self, dim: int = 48):
        super().__init__()
        self.squeezing = nn.Sequential(nn.Conv2d(dim, 1, 1, 1, 0), nn.Hardsigmoid(True))

    def forward(self, x):
        return x * self.squeezing(x)


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
    ----
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, in_dim: int = 64, num_feat: int = 64, out_dim: int = 3, scale: int = 4):
        m = [nn.Conv2d(in_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)]
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        m.append(nn.Conv2d(num_feat, out_dim, 3, 1, 1))
        super(Upsample, self).__init__(*m)


class Interpolate(nn.Module):
    def __init__(self, scale_factor: int = 4, mode: str = 'nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class InterpolateUpsampler(nn.Sequential):
    def __init__(self, dim: int = 64, out_ch: int = 3, scale: int = 4):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                m.extend(
                    (
                        nn.Conv2d(dim, dim, 3, 1, 1),
                        Interpolate(2),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            m.extend(
                (
                    nn.Conv2d(dim, dim, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
        elif scale == 3:
            m.extend(
                (
                    nn.Conv2d(dim, dim, 3, 1, 1),
                    Interpolate(scale),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(dim, dim, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

        m.append(nn.Conv2d(dim, out_ch, 3, 1, 1))
        super().__init__(*m)


class CFFT(nn.Module):
    def __init__(self, dim: int = 64, expansion_factor: float = 1.5):
        super(CFFT, self).__init__()

        self.dim = dim
        self.expansion_factor = expansion_factor

        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=hidden_features,
            bias=True,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=True)
        self.act = nn.Mish(False)

    def forward(self, x):
        x = self.project_in(x)
        x = self.act(x)
        x = channel_shuffle(x, 2)
        x = self.dwconv(x) + x
        return self.project_out(x)


class LayerNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, dim, expansion_esa=0.25):
        super(ESA, self).__init__()
        f = int(dim * expansion_esa)
        self.conv1 = nn.Conv2d(dim, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, dim, kernel_size=1)
        self.sigmoid = nn.Hardsigmoid(True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _B, _C, H, W = x.shape
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (H, W), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


def channel_shuffle(x, groups: int):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, num_channels, height, width)
    return x


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 1.5,
        conv_ratio: float = 1.0,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.norm = LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)

        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size,
            1,
            kernel_size // 2,
            groups=conv_channels,
        )
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return (x * self.gamma) + shortcut


class Block(nn.Module):
    def __init__(self, dim: int = 64, expansion_factor: float = 1.5):
        super().__init__()
        self.token_mix = GatedCNNBlock(dim, expansion_factor)
        self.ffn = nn.Sequential(LayerNorm(dim, eps=1e-6), CFFT(dim, expansion_factor))
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.se = SSELayer(dim)

    def forward(self, x):
        x = self.token_mix(x) + x
        x = self.gamma * self.ffn(x) + x
        return self.se(x)


class Blocks(nn.Module):
    def __init__(self, dim: int = 64, blocks: int = 4, expansion_factor: float = 1.5, expansion_esa: float = 0.25):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(dim, expansion_factor) for _ in range(blocks)] + [ESA(dim, expansion_esa)])
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        return self.blocks(x) * self.gamma + x


class MoESR(nn.Module):
    """Mamba out Excitation Super-Resolution"""

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        scale: int = 4,
        dim: int = 64,
        n_blocks: int = 6,
        n_block: int = 6,
        expansion_factor: int = 1.5,
        expansion_esa: int = 0.25,
        upsampler: Literal['n+c', 'psd', 'ps', 'dys', 'conv'] = 'n+c',
        upsample_dim: int = 64,
    ):
        super().__init__()

        self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)
        self.blocks = nn.Sequential(*[Blocks(dim, n_block, expansion_factor, expansion_esa) for _ in range(n_blocks)])
        if upsampler == 'n+c':
            self.upscale = InterpolateUpsampler(dim, out_ch, scale)
        elif upsampler == 'psd':
            self.upscale = nn.Sequential(nn.Conv2d(64, 3 * 4 * 4, 3, 1, 1), nn.PixelShuffle(scale))
        elif upsampler == 'ps':
            self.upscale = Upsample(dim, upsample_dim, out_ch, scale)
        elif upsampler == 'dys':
            self.upscale = DySample(dim, out_ch, scale)
        elif upsampler == 'conv':
            self.upscale = nn.Conv2d(dim, out_ch, 3, 1, 1)
        self.register_buffer('metadata', torch.tensor([in_ch, out_ch, scale], dtype=torch.uint8))
        self.gamma = nn.Parameter(torch.ones(1, 64, 1, 1), requires_grad=True)

    def forward(self, x):
        x = self.in_to_dim(x)
        x = self.blocks(x) * self.gamma + x
        return self.upscale(x)
