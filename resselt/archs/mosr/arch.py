import torch

from torch import nn
from torch.nn.init import trunc_normal_


# upscale, __ = net_opt()
class DCCM(nn.Sequential):
    "Doubled Convolutional Channel Mixer"

    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class LKAT(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, 1, 0), nn.GELU())
        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),
            nn.Conv2d(
                n_feats,
                n_feats,
                9,
                stride=1,
                padding=(9 // 2) * 3,
                groups=n_feats,
                dilation=3,
            ),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
        )

        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.att(x)
        return self.conv1(x)


class GatedCNNBlock(nn.Module):
    r"""Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(self, dim, expansion_ratio=8 / 3, kernel_size=7, conv_ratio=1.0, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=conv_channels)
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
    def __init__(self, dim, drop_path, expansion_ratio):
        super().__init__()
        self.dccm = DCCM(dim)
        self.gcnn = GatedCNNBlock(dim, expansion_ratio=expansion_ratio, drop_path=drop_path)
        # self.cab =LKAT(dim)

    def forward(self, x):
        x = self.dccm(x)
        x = self.gcnn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x = self.cab(x)
        return x


class GBlocks(nn.Module):
    def __init__(self, n_blocks, dim, drop_path, expansion_ratio):
        super().__init__()
        self.gated = nn.Sequential(*[GatedBlocks(dim, drop_path, expansion_ratio) for _ in range(n_blocks)])
        self.end = LKAT(dim)

    def forward(self, x):
        shortcut = x
        x = self.gated(x)
        return self.end(x) + shortcut


class mosr(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        upscale: int = 2,
        blocks: list[int] = [5],
        dim: int = 48,
        upsampler: str = 'ps',
        drop_path: float = 0.0,
        is_lite: bool = False,
        expansion_ratio: float = 1.5,
    ):
        super(mosr, self).__init__()

        # dp_rates = [x.item() for x in torch.linspace(0, drop_path, blocks)]
        self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)
        self.gblocks = nn.Sequential(*[GBlocks(block, dim, 0, expansion_ratio=expansion_ratio) for block in blocks])
        if upsampler == 'ps':
            self.upsampler = nn.Sequential(nn.Conv2d(dim, out_ch * (upscale**2), 3, padding=1), nn.PixelShuffle(upscale))
            # trunc_normal_(self.upsampler[0].weight, std=0.02)
        elif upsampler == 'dys':
            # self.upsampler = DySample(dim, out_ch, upscale)
            pass
        elif upsampler == 'conv':
            if upsampler != 1:
                msg = 'conv supports only 1x'
                raise ValueError(msg)

            self.upsampler = nn.Conv2d(dim, out_ch, 3, padding=1)
        else:
            raise NotImplementedError(
                f'upsampler: {upsampler} not supported, choose one of these options: \
                ["ps", "dys", "conv"] conv supports only 1x'
            )

    def forward(self, x):
        x = self.in_to_dim(x)
        x = self.gblocks(x)
        return self.upsampler(x)


if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.path.abspath('.'))
    upscale = 4
    height = 256
    width = 256
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = mosr().cuda()

    params = sum(map(lambda x: x.numel(), model.parameters()))
    results = dict()
    results['runtime'] = []
    model.eval()

    x = torch.randn((1, 3, height, width)).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # x = model(x)
        for _ in range(10):
            x_sr = model(x)
        for _ in range(100):
            start.record()
            x_sr = model(x)
            end.record()
            torch.cuda.synchronize()
            results['runtime'].append(start.elapsed_time(end))  # milliseconds
    print(x.shape)

    print('{:.2f}ms'.format(sum(results['runtime']) / len(results['runtime'])))
    results['memory'] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024**2
    print('Max Memery:{:.2f}[M]'.format(results['memory']))
    print(f'Height:{height}->{x_sr.shape[2]}\nWidth:{width}->{x_sr.shape[3]}\nParameters:{params / 1e3:.2f}K')
