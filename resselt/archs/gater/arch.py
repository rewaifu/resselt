import math
import torch.nn.functional as F  # noqa: N812
from torch import nn
import torch


class DConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, padding=7 // 2, groups=dim)

    def forward(self, x, res):
        B, N, C = x.shape
        H, W = res
        x = x.transpose(1, 2).view(B, C, H, W)
        return self.conv(x).flatten(2).transpose(1, 2)


class FLPVT2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.focusing_factor = nn.Parameter(torch.ones(dim) * focusing_factor)
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size, groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(dim))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.RMSNorm):
            # nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, resol):
        H, W = resol
        B, N, C = x.shape
        q = self.q(x)

        kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]
        scale = torch.functional.F.softplus(self.scale)
        q = torch.functional.F.relu(q) + 1e-6
        k = torch.functional.F.relu(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q**self.focusing_factor
        k = k**self.focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n**-0.5)) @ (v * (n**-0.5))
        x = q @ kv * z
        x = x.transpose(1, 2).reshape(B, N, C)
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = self.proj(x)

        return x


class GatedCNNBlock(nn.Module):
    r"""Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(self, dim, expansion_ratio=8 / 3, conv_ratio=1.0, drop_path=0.0, att=False):
        super().__init__()
        if att:
            expansion_ratio = 1.5
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = FLPVT2(conv_channels) if att else DConv(conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, res):
        # shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = self.conv(c, res)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        return x  # + shortcut


class Blocks(nn.Module):
    def __init__(self, dim, n_block, att=False):
        super().__init__()
        self.gated = nn.ModuleList([GatedCNNBlock(dim, att=att) for _ in range(n_block)])

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for block in self.gated:
            x = block(x, (H, W)) + x
        return x.transpose(1, 2).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=True), nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=True), nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class GateR(nn.Module):
    def __init__(self, dim=48, in_ch=3, num_blocks=(3, 6, 6, 10, 6, 6, 3), latent_att=False):
        super().__init__()
        self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)

        self.enc0 = Blocks(dim, num_blocks[0])
        self.enc1 = nn.Sequential(Downsample(dim), Blocks(dim * 2, num_blocks[1]))
        self.enc2 = nn.Sequential(Downsample(dim * 2), Blocks(dim * 4, num_blocks[2]))

        self.latent = nn.Sequential(Downsample(dim * 4), Blocks(dim * 8, num_blocks[3], latent_att), Upsample(dim * 8))

        self.dec0 = nn.Sequential(nn.Conv2d(dim * 8, dim * 4, 1), Blocks(dim * 4, num_blocks[4]), Upsample(dim * 4))
        self.dec1 = nn.Sequential(nn.Conv2d(dim * 4, dim * 2, 1), Blocks(dim * 2, num_blocks[5]), Upsample(dim * 2))
        self.dec2 = nn.Sequential(Blocks(dim * 2, num_blocks[6]))
        self.dim_to_ch = nn.Sequential(nn.Conv2d(dim * 2, dim, 3, 1, 1), nn.Conv2d(dim, in_ch, 3, 1, 1))

        # self.dec0 =

    def check_img_size(self, x, resolution: tuple[int, int]):
        h, w = resolution
        scaled_size = 8
        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.check_img_size(x, (H, W))
        enc = self.in_to_dim(x)
        enc0 = self.enc0(enc)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)

        latent = self.latent(enc2)

        dec0 = self.dec0(torch.cat([latent, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec0, enc1], dim=1))
        dec2 = self.dec2(torch.cat([dec1, enc0], dim=1))
        return (self.dim_to_ch(dec2) + x)[:, :, :H, :W]
