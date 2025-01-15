import math
from typing import Literal

import numpy as np
import torch.nn.functional as F
import torch
from torch import nn, Tensor


SampleMods = Literal['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample']


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
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
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

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


class UniUpsample(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods,
        scale: int = 2,
        in_dim: int = 64,
        out_dim: int = 3,
        mid_dim: int = 64,  # Only pixelshuffle
        group: int = 4,  # Only DySample
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
            m.append(DySample(in_dim, out_dim, scale, group))
        else:
            raise ValueError(f'An invalid Upsample was selected. Please choose one of {SampleMods}')
        super().__init__(*m)

        self.register_buffer(
            'MetaUpsample',
            torch.tensor(
                [
                    1,  # Block version, if you change something, please number from the end so that you can distinguish between authorized changes and third parties
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


class LayerNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LinAngularAttention(nn.Module):
    def __init__(
        self,
        in_channels=64,
        num_heads=8,
        qkv_bias=True,
        res_kernel_size=9,
    ):
        super().__init__()
        assert in_channels % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(in_channels, in_channels)
        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, x):
        N, C, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        L = h * w
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.training:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02  # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v)

        attn = k.transpose(-2, -1) @ v

        if self.training:
            x = sparse @ v + 0.5 * v + 1.0 / torch.pi * q @ attn
        else:
            x = 0.5 * v + 1.0 / torch.pi * q @ attn
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        return x.permute(0, 2, 1).reshape(N, C, h, w)


class OmniShift(nn.Module):
    def __init__(self, dim: int = 48) -> None:
        super().__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=True)
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=True,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=True,
        )
        self.alpha1 = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        self.alpha3 = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        self.alpha4 = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=True,
        )

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        out = self.alpha1 * x + self.alpha2 * out1x1 + self.alpha3 * out3x3 + self.alpha4 * out5x5
        return out

    def reparam_5x5(self) -> None:
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))

        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))
        combined_weight = (
            self.alpha1.transpose(0, 1) * identity_weight
            + self.alpha2.transpose(0, 1) * padded_weight_1x1
            + self.alpha3.transpose(0, 1) * padded_weight_3x3
            + self.alpha4.transpose(0, 1) * self.conv5x5.weight
        )

        combined_bias = (
            self.alpha2.squeeze() * self.conv1x1.bias + self.alpha3.squeeze() * self.conv3x3.bias + self.alpha4.squeeze() * self.conv5x5.bias
        )

        device = self.conv5x5_reparam.weight.device

        combined_weight = combined_weight.to(device)
        combined_bias = combined_bias.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)
        self.conv5x5_reparam.bias = nn.Parameter(combined_bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.reparam_5x5()

    def forward(self, x):
        if self.training:
            out = self.forward_train(x)
        else:
            out = self.conv5x5_reparam(x)
        return out


class HybridAttention(nn.Module):
    def __init__(self, dim, down):
        super().__init__()
        self.att = nn.Sequential(nn.MaxPool2d(down, down), OmniShift(dim // 2), nn.Upsample(scale_factor=down, mode='bilinear'))
        self.conv = OmniShift(dim // 2)
        self.aggr = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Mish(True))

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv(x1)
        x2 = self.att(x2) + x1
        return self.aggr(torch.cat([x1, x2], dim=1)) * x


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
        down: int = 1,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)
        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = HybridAttention(dim, down)
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return x + shortcut


class GatedGroup(nn.Module):
    def __init__(self, res_blocks, dim, down_sample, expansion_ratio):
        super().__init__()
        self.register_buffer('down_sample', torch.tensor(down_sample, dtype=torch.uint8))
        self.body = nn.Sequential(
            *[GatedCNNBlock(dim, down=down_sample, expansion_ratio=expansion_ratio) for _ in range(res_blocks)] + [HybridAttention(dim, down_sample)]
        )

    def forward(self, x):
        return self.body(x) + x


class RHA(nn.Module):
    """Residual Hybrid Attention"""

    def __init__(
        self,
        dim=64,
        scale=4,
        in_ch=3,
        out_ch=3,
        mid_dim=64,
        down_list=(8, 4),
        expansion_ratio=1.5,
        group_blocks=4,
        res_blocks=6,
        upsample: SampleMods = 'pixelshuffledirect',
        unshuffle_mod=False,
    ):
        super().__init__()
        unshuffle = 0
        if scale < 4 and unshuffle_mod:
            if scale == 3:
                raise ValueError('Unshuffle_mod does not support 3x')
            unshuffle = 4 // scale
            self.register_buffer('unshuffle', torch.tensor(unshuffle, dtype=torch.uint8))
            scale = 4
        self.pad = unshuffle if unshuffle > 0 else 1
        self.pad *= np.max(down_list)
        self.to_feat = (
            nn.Sequential(nn.PixelUnshuffle(unshuffle), nn.Conv2d(in_ch * unshuffle**2, dim, 3, 1, 1))
            if unshuffle_mod
            else (nn.Conv2d(in_ch, dim, 3, 1, 1))
        )
        down_sample_len = len(down_list)

        self.body = nn.Sequential(*[GatedGroup(res_blocks, dim, down_list[i % down_sample_len], expansion_ratio) for i in range(group_blocks)])
        self.scale = scale
        self.to_img = UniUpsample(upsample, scale, dim, out_ch, mid_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_img_size(self, x, resolution: tuple[int, int]):
        scaled_size = self.pad
        mod_pad_h = (scaled_size - resolution[0] % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - resolution[1] % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.check_img_size(x, (h, w))
        x = self.to_feat(x)
        x = self.body(x) + x
        return self.to_img(x)[:, :, : h * self.scale, : w * self.scale]
