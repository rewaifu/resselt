import torch
from torch import nn

from resselt.archs.utils import DySample


class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(self, dim,
                 expansion_ratio=8 / 3,
                 kernel_size=7,
                 conv_ratio=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                              groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        return x + shortcut


class GatedBlocks(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_blocks,
                 expansion_ratio
                 ):
        super().__init__()
        self.in_to_out = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.gcnn = nn.Sequential(
            *[GatedCNNBlock(out_dim,
                            expansion_ratio=expansion_ratio)
              for _ in range(n_blocks)
              ])

    def forward(self, x):
        x = self.in_to_out(x)
        short_cut = x
        x = x.permute(0, 2, 3, 1)
        x = self.gcnn(x)
        return x.permute(0, 3, 1, 2) + short_cut


class MoSR(nn.Module):
    def __init__(self,
                 in_ch: int = 3,
                 out_ch: int = 3,
                 upscale: int = 4,
                 blocks: tuple[int] = (6, 9, 9),
                 dims: tuple[int] = (64, 96, 96),
                 upsampler: str = "ps",
                 expansion_ratio: float = 1.0
                 ):
        super(MoSR, self).__init__()
        len_blocks = len(blocks)
        dims = [in_ch] + list(dims)
        self.gblocks = nn.Sequential(
            *[GatedBlocks(dims[i], dims[i + 1], blocks[i],
                          expansion_ratio=expansion_ratio
                          )
              for i in range(len_blocks)]
        )

        if upsampler == "ps":
            self.upsampler = nn.Sequential(
                nn.Conv2d(dims[-1],
                          out_ch * (upscale ** 2),
                          3, padding=1),
                nn.PixelShuffle(upscale)
            )
        elif upsampler == "dys":
            self.upsampler = DySample(dims[-1], out_ch, upscale)
        elif upsampler == "conv":
            if upscale != 1:
                msg = "conv supports only 1x"
                raise ValueError(msg)

            self.upsampler = nn.Conv2d(dims[-1],
                                       out_ch,
                                       3, padding=1)
        else:
            raise NotImplementedError(
                f'upsampler: {upsampler} not supported, choose one of these options: \
                ["ps", "dys", "conv"] conv supports only 1x')

    def forward(self, x):
        x = self.gblocks(x)
        return self.upsampler(x)
