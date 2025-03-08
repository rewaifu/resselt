import math
from typing import Literal

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


def l2_norm(x):
    return torch.einsum('bcn, bn->bcn', x, 1 / torch.norm(x, p=2, dim=-2))


class Attention(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6) -> None:
        super().__init__()
        # self.gamma = nn.Parameter(torch.zeros(1,in_places,1,1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum('bnc, bc->bn', Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum('bcn->bc', V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum('bnm, bmc->bcn', Q, matrix)

        weight_value = torch.einsum('bcn, bn->bcn', matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        return weight_value.contiguous()


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


class ConvBlock(nn.Module):
    r"""https://github.com/joshyZhou/AST/blob/main/model.py#L22"""

    def __init__(self, in_channel: int, out_channel: int, strides: int = 1) -> None:
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.Mish(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.Mish(True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class GateRV2(nn.Module):
    def __init__(
        self,
        in_ch=3,
        dim=48,
        enc_blocks=(2, 2, 4, 6),
        dec_blocks=(2, 2, 2, 2),
        num_latent=10,
        scale=1,
        upsample: SampleMods = 'pixelshuffledirect',
        upsample_mid_dim=32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.scale = 1
        self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)

        self.encode = nn.ModuleList([Block(dim * (2**i), enc_blocks[i]) for i in range(len(enc_blocks))])
        self.latent = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim * (2 ** len(enc_blocks)),
                    expansion_ratio=1.5,
                    conv_ratio=1.00,
                    att=True,
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

        if scale != 1:
            self.dim_to_in = nn.Identity()
            self.short_to_dim = ConvBlock(in_ch, dim)
            self.upsample = UniUpsample(upsample, scale, dim, in_ch, upsample_mid_dim)
        else:
            self.dim_to_in = nn.Conv2d(dim, in_ch, 3, 1, 1)
            self.short_to_dim = nn.Identity()
            self.upsample = nn.Identity()

    def load_state_dict(self, state_dict, *args, **kwargs):
        if 'upsample.MetaUpsample' in state_dict:
            state_dict['upsample.MetaUpsample'] = self.upsample.MetaUpsample
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
        shorts = []
        for block in self.encode:
            x, short = block(x)
            shorts.append(short)
        x = self.latent(x)
        len_block = len(self.decode)
        shorts.reverse()
        for index in range(len_block):
            x = self.decode[index](x, shorts[index])
        x = self.dim_to_in(x) + self.short_to_dim(inp)

        return self.upsample(x)[:, :, : H * self.scale, : W * self.scale]
