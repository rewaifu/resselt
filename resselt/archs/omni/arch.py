from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import einsum, nn

import torch
import torch.nn.functional as F

from ...utilities import pad_to_multiple


def moment(x, dim=(2, 3), k=2):
    assert len(x.size()) == 4
    mean = torch.mean(x, dim=dim).unsqueeze(-1).unsqueeze(-1)
    mk = (1 / (x.size(2) * x.size(3))) * torch.sum(torch.pow(x - mean, k), dim=dim)
    return mk


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super().__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class LK_ESA(nn.Module):
    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d, kernel_expand=1, bias=True):
        super().__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)

        kernel_size = 17
        kernel_expand = kernel_expand
        padding = kernel_size // 2

        self.vec_conv = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=2,
            bias=bias,
        )
        self.vec_conv3x1 = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=2,
            bias=bias,
        )

        self.hor_conv = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=2,
            bias=bias,
        )
        self.hor_conv1x3 = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=2,
            bias=bias,
        )

        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)

        res = self.vec_conv(c1_) + self.vec_conv3x1(c1_)
        res = self.hor_conv(res) + self.hor_conv1x3(res)

        cf = self.conv_f(c1_)
        c4 = self.conv4(res + cf)
        m = self.sigmoid(c4)
        return x * m


class LK_ESA_LN(nn.Module):
    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d, kernel_expand=1, bias=True):
        super().__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)

        kernel_size = 17
        kernel_expand = kernel_expand
        padding = kernel_size // 2

        self.norm = LayerNorm2d(n_feats)

        self.vec_conv = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=2,
            bias=bias,
        )
        self.vec_conv3x1 = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=2,
            bias=bias,
        )

        self.hor_conv = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=2,
            bias=bias,
        )
        self.hor_conv1x3 = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=2,
            bias=bias,
        )

        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.norm(x)
        c1_ = self.conv1(c1_)

        res = self.vec_conv(c1_) + self.vec_conv3x1(c1_)
        res = self.hor_conv(res) + self.hor_conv1x3(res)

        cf = self.conv_f(c1_)
        c4 = self.conv4(res + cf)
        m = self.sigmoid(c4)
        return x * m


class AdaGuidedFilter(nn.Module):
    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d, kernel_expand=1, bias=True):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(
            in_channels=n_feats,
            out_channels=1,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.r = 5

    def box_filter(self, x, r):
        channel = x.shape[1]
        kernel_size = 2 * r + 1
        weight = 1.0 / (kernel_size**2)
        box_kernel = weight * torch.ones((channel, 1, kernel_size, kernel_size), dtype=torch.float32, device=x.device)
        output = F.conv2d(x, weight=box_kernel, stride=1, padding=r, groups=channel)
        return output

    def forward(self, x):
        _, _, H, W = x.shape
        N = self.box_filter(torch.ones((1, 1, H, W), dtype=x.dtype, device=x.device), self.r)

        # epsilon = self.fc(self.gap(x))
        # epsilon = torch.pow(epsilon, 2)
        epsilon = 1e-2

        mean_x = self.box_filter(x, self.r) / N
        var_x = self.box_filter(x * x, self.r) / N - mean_x * mean_x

        A = var_x / (var_x + epsilon)
        b = (1 - A) * mean_x
        m = A * x + b

        # mean_A = self.box_filter(A, self.r) / N
        # mean_b = self.box_filter(b, self.r) / N
        # m = mean_A * x + mean_b
        return x * m


class AdaConvGuidedFilter(nn.Module):
    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d, kernel_expand=1, bias=True):
        super().__init__()
        f = esa_channels

        self.conv_f = conv(f, f, kernel_size=1)

        kernel_size = 17
        kernel_expand = kernel_expand
        padding = kernel_size // 2

        self.vec_conv = nn.Conv2d(
            in_channels=f,
            out_channels=f,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=f,
            bias=bias,
        )

        self.hor_conv = nn.Conv2d(
            in_channels=f,
            out_channels=f,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=f,
            bias=bias,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(
            in_channels=f,
            out_channels=f,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

    def forward(self, x):
        y = self.vec_conv(x)
        y = self.hor_conv(y)

        sigma = torch.pow(y, 2)
        epsilon = self.fc(self.gap(y))

        weight = sigma / (sigma + epsilon)

        m = weight * x + (1 - weight)

        return x * m


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, bias=False):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(
        in_channels,
        out_channels * (upscale_factor**2),
        kernel_size,
        padding=1,
        bias=bias,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# helper classes


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1, 1, 0),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1, 1, 0),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.0):
        super().__init__()

        hidden_features = int(dim * mult)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# MBConv


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1'),
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


def MBConv(dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        # nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.0,
        window_size=7,
        with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))

        # relative positional bias
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = (  # noqa: F841
            *x.shape,
            x.device,
            self.heads,
        )

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = (rearrange(t, 'b n (h d ) -> b h n d', h=h) for t in (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)


class Block_Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        bias=False,
        dropout=0.0,
        window_size=7,
        with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.ps = window_size
        self.scale = dim_head**-0.5
        self.with_pe = with_pe

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # project for queries, keys, values
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # split heads

        q, k, v = (
            rearrange(
                t,
                'b (h d) (x w1) (y w2) -> (b x y) h (w1 w2) d',
                h=self.heads,
                w1=self.ps,
                w2=self.ps,
            )
            for t in (q, k, v)
        )

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(
            out,
            '(b x y) head (w1 w2) d -> b (head d) (x w1) (y w2)',
            x=h // self.ps,
            y=w // self.ps,
            head=self.heads,
            w1=self.ps,
            w2=self.ps,
        )

        out = self.to_out(out)
        return out


class Channel_Attention(nn.Module):
    def __init__(self, dim, heads, bias=False, dropout=0.0, window_size=7):
        super().__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = (
            rearrange(
                t,
                'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)',
                ph=self.ps,
                pw=self.ps,
                head=self.heads,
            )
            for t in qkv
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out,
            'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)',
            h=h // self.ps,
            w=w // self.ps,
            ph=self.ps,
            pw=self.ps,
            head=self.heads,
        )

        out = self.project_out(out)

        return out


class Channel_Attention_grid(nn.Module):
    def __init__(self, dim, heads, bias=False, dropout=0.0, window_size=7):
        super().__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = (
            rearrange(
                t,
                'b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)',
                ph=self.ps,
                pw=self.ps,
                head=self.heads,
            )
            for t in qkv
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out,
            'b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)',
            h=h // self.ps,
            w=w // self.ps,
            ph=self.ps,
            pw=self.ps,
            head=self.heads,
        )

        out = self.project_out(out)

        return out


class OSA_Block(nn.Module):
    def __init__(
        self,
        channel_num=64,
        window_size=8,
        with_pe=False,
        dropout=0.0,
    ):
        super().__init__()

        w = window_size

        self.layer = nn.Sequential(
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25,
            ),
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # block-like attention
            PreNormResidual(
                channel_num,
                Attention(
                    dim=channel_num,
                    dim_head=channel_num // 4,
                    dropout=dropout,
                    window_size=window_size,
                    with_pe=with_pe,
                ),
            ),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # channel-like attention
            Conv_PreNormResidual(
                channel_num,
                Channel_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size),
            ),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),  # grid-like attention
            PreNormResidual(
                channel_num,
                Attention(
                    dim=channel_num,
                    dim_head=channel_num // 4,
                    dropout=dropout,
                    window_size=window_size,
                    with_pe=with_pe,
                ),
            ),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # channel-like attention
            Conv_PreNormResidual(
                channel_num,
                Channel_Attention_grid(dim=channel_num, heads=4, dropout=dropout, window_size=window_size),
            ),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class OSAG(nn.Module):
    def __init__(
        self,
        channel_num=64,
        bias=True,
        block_num=4,
        window_size=0,
        pe=False,
    ):
        super().__init__()

        # print("window_size: %d" % (window_size))
        # print("with_pe", pe)
        # print("ffn_bias: %d" % (ffn_bias))

        # block_script_name = kwargs.get("block_script_name", "OSA")
        # block_class_name = kwargs.get("block_class_name", "OSA_Block")

        # script_name = "." + block_script_name
        # package = __import__(script_name, fromlist=True)
        block_class = OSA_Block  # getattr(package, block_class_name)
        group_list = []
        for _ in range(block_num):
            temp_res = block_class(
                channel_num,
                window_size=window_size,
                with_pe=pe,
            )
            group_list.append(temp_res)
        group_list.append(nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias))
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)

    def forward(self, x):
        out = self.residual_layer(x)
        out = out + x
        return self.esa(out)


class OmniSR(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        block_num=1,
        pe=True,
        window_size=8,
        res_num=1,
        up_scale=4,
        bias=True,
    ):
        super().__init__()

        residual_layer = []
        self.res_num = res_num

        self.up_scale = up_scale
        self.window_size = window_size

        for _ in range(res_num):
            temp_res = OSAG(
                channel_num=num_feat,
                bias=bias,
                block_num=block_num,
                window_size=self.window_size,
                pe=pe,
            )
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(
            in_channels=num_in_ch,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.output = nn.Conv2d(
            in_channels=num_feat,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

    def check_image_size(self, x):
        return pad_to_multiple(x, self.window_size, mode='constant')

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = out[:, :, : H * self.up_scale, : W * self.up_scale]
        return out
