import math
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
            if mid_dim != in_dim:
                m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
                dys_dim = mid_dim
            else:
                dys_dim = in_dim
            m.append(DySample(dys_dim, out_dim, scale, group))
        else:
            raise ValueError(f'An invalid Upsample was selected. Please choose one of {SampleMods}')
        super().__init__(*m)

        self.register_buffer(
            'MetaUpsample',
            torch.tensor(
                [
                    2,  # Block version, if you change something, please number from the end so that you can distinguish between authorized changes and third parties
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

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class FocusedLinearAttention(nn.Module):
    r"""https://github.com/LeapLabTHU/FLatten-Transformer/blob/master/models/flatten_swin.py
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window."""

    def __init__(
        self,
        dim: int = 64,
        window_size: int = 8,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        focusing_factor: int = 3,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=kernel_size,
            groups=head_dim,
            padding=kernel_size // 2,
        )
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size**2, dim)))

    def window_partition(self, x: Tensor) -> (Tensor, int, int, int):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, C, H, W = x.shape
        x = x.view(
            B,
            C,
            H // self.window_size,
            self.window_size,
            W // self.window_size,
            self.window_size,
        )
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows, H, W, C

    def window_reverse(self, windows: Tensor, h: int, w: int) -> Tensor:
        B = int(windows.shape[0] / (h * w / self.window_size / self.window_size))
        x = windows.view(
            B,
            h // self.window_size,
            w // self.window_size,
            self.window_size,
            self.window_size,
            -1,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, h, w)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)-
        """

        x, h, w, C = self.window_partition(x)  # nW*B, window_size, window_size, C
        x = x.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        k = k + self.positional_encoding
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6
        scale = F.softplus(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q**self.focusing_factor
        k = k**self.focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (N**-0.5)) @ (v * (N**-0.5))
        x = q @ kv * z

        H = W = self.window_size
        x = x.transpose(1, 2).reshape(B, N, C)
        v = v.reshape(B * self.num_heads, H, W, self.head_dim).permute(0, 3, 1, 2)
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.window_reverse(x, h, w)
        return x


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

    def forward_train(self, x: Tensor) -> Tensor:
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
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            out = self.forward_train(x)
        else:
            out = self.conv5x5_reparam(x)
        return out


class Shift(nn.Module):
    def __init__(self, shift: int = 4) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: Tensor) -> Tensor:
        return torch.roll(x, shifts=(self.shift, self.shift), dims=(2, 3))


class HybridAttention(nn.Module):
    def __init__(self, dim: int = 64, down: int = 8, shift: int = 0, window_size: int = 8) -> None:
        super().__init__()
        self.att = self.att = nn.Sequential(
            nn.MaxPool2d(down, down) if down > 1 else nn.Identity(),
            Shift(-shift) if shift else nn.Identity(),
            FocusedLinearAttention(dim // 2, window_size),
            Shift(shift) if shift else nn.Identity(),
            nn.Upsample(scale_factor=down, mode='bilinear') if down > 1 else nn.Identity(),
        )
        self.conv = OmniShift(dim // 2)
        self.aggr = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Mish(True))

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv(x1)
        x2 = self.att(x2)
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
        shift: int = 0,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)
        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = HybridAttention(dim, down, shift, window_size)
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return x + shortcut


class GatedGroup(nn.Module):
    def __init__(
        self,
        res_blocks: int = 6,
        dim: int = 64,
        down_sample: int = 4,
        expansion_ratio: float = 1.5,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.register_buffer('down_sample', torch.tensor(down_sample, dtype=torch.uint8))

        self.body = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim,
                    down=down_sample,
                    expansion_ratio=expansion_ratio,
                    shift=0 if (i % 2 == 0) else window_size // 2,
                    window_size=window_size,
                )
                for i in range(res_blocks)
            ]
            + [OmniShift(dim), nn.Conv2d(dim, dim, kernel_size=1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x) + x


class RHA(nn.Module):
    """Residual Hybrid Attention"""

    def __init__(
        self,
        dim: int = 64,
        scale: int = 4,
        in_ch: int = 3,
        out_ch: int = 3,
        mid_dim: int = 32,
        down_list: Sequence[int] = (
            8,
            4,
        ),
        expansion_ratio: float = 1.5,
        group_blocks: int = 4,
        res_blocks: int = 6,
        upsample: SampleMods = 'pixelshuffledirect',
        unshuffle_mod: bool = False,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        unshuffle = 0
        if scale < 4 and unshuffle_mod:
            if scale == 3:
                raise ValueError('Unshuffle_mod does not support 3x')
            unshuffle = 4 // scale
            self.register_buffer('unshuffle', torch.tensor(unshuffle, dtype=torch.uint8))
            scale = 4

        self.pad: int = unshuffle if unshuffle > 0 else 1
        self.pad *= np.max(down_list) * window_size
        self.to_feat = (
            nn.Sequential(
                nn.PixelUnshuffle(unshuffle),
                nn.Conv2d(in_ch * unshuffle**2, dim, 3, 1, 1),
            )
            if unshuffle_mod
            else (nn.Conv2d(in_ch, dim, 3, 1, 1))
        )

        down_sample_len = len(down_list)
        self.body = nn.Sequential(
            *[
                GatedGroup(
                    res_blocks,
                    dim,
                    down_list[i % down_sample_len],
                    expansion_ratio,
                    window_size,
                )
                for i in range(group_blocks)
            ]
        )
        self.scale = scale
        self.to_img = UniUpsample(upsample, scale, dim, out_ch, mid_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict['to_img.MetaUpsample'] = self.to_img.MetaUpsample
        if 'unshuffle' in state_dict:
            state_dict['unshuffle'] = self.unshuffle
        return super().load_state_dict(state_dict, *args, **kwargs)

    def check_img_size(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        scaled_size = self.pad
        mod_pad_h = (scaled_size - resolution[0] % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - resolution[1] % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.check_img_size(x, (h, w))
        x = self.to_feat(x)
        x = self.body(x) + x
        return self.to_img(x)[:, :, : h * self.scale, : w * self.scale]
