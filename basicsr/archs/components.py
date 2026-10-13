import math
import numbers

import torch
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from basicsr.archs.kpn_pixel import IDynamicDWConv
from basicsr.archs.mamba import SS2D
from basicsr.archs.scan import ScanTransform


class SimpleGate(nn.Module):
    def __init__(self) -> None:
        super(SimpleGate, self).__init__()
        self.dim = 1

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2


class SimpleGate2(nn.Module):
    def __init__(self) -> None:
        super(SimpleGate2, self).__init__()
        self.dim = 1

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * torch.sigmoid(x2)


class RepConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, bias=False) -> None:
        super(RepConv, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, bias=bias, padding=kernel_size // 2, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, bias=bias, padding=1, groups=channels)

    def forward(self, x: Tensor) -> Tensor:
        res_1 = self.conv1(x)
        res_2 = self.conv2(x)
        return res_1 + res_2 + x


class RepConv2(nn.Module):
    def __init__(self, channels: int, bias=False) -> None:
        super(RepConv2, self).__init__()
        self.conv2 = nn.Conv2d(channels, channels, 3, bias=bias, padding=1, groups=channels)

    def forward(self, x: Tensor) -> Tensor:
        res_2 = self.conv2(x)
        return res_2 + x


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        x = self.down(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


def flow_warp(x: Tensor, flow: Tensor, interpolation='bilinear', padding_mode='zeros', align_corners=True) -> Tensor:
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x,
                           grid_flow,
                           mode=interpolation,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return output


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DownSampling(nn.Module):
    """
    PixelUnshuffle 下采样两倍
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DownSampling, self).__init__()

        self.conv = nn.Conv2d(in_channels * 2 * 2, out_channels, kernel_size=1)
        self.down = nn.PixelUnshuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        x = self.conv(x)
        return x


class UpSampling(nn.Module):
    """
    PixelUnshuffle 上采样两倍
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UpSampling, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels * 2 * 2, kernel_size=1)
        self.down = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.down(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResBlock, self).__init__()

        self.conv_in = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.conv_in(x)
        x = self.act(x)
        x = self.conv_out(x)

        return x + res


class CALayer(nn.Module):
    def __init__(self, channels: int, bias=False) -> None:
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction = 1
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    """
    Channel Attention Block
    """

    def __init__(self, channels: int, reduction: int, add_channels: int = 0) -> None:
        super(CAB, self).__init__()

        if add_channels != 0:
            in_channels = channels + add_channels
        else:
            in_channels = channels

        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, channels * 2, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.Atten = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.Conv[0](x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = x1 * x2
        x = self.Conv[1](x)
        x = x * self.Atten(x)
        return x + res


class CAB1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, bias=False) -> None:
        super(CAB1, self).__init__()
        modules_body = []
        scale_factor = 1
        n_scale_feat = int(scale_factor * channels)
        self.norm = LayerNorm(channels)
        modules_body.append(nn.Conv2d(channels, n_scale_feat * 2, 1, bias=bias))
        modules_body.append(RepConv2(n_scale_feat * 2, bias))
        modules_body.append(SimpleGate())
        modules_body.append(RepConv(n_scale_feat, kernel_size, bias))
        modules_body.append(nn.Conv2d(n_scale_feat, 2 * n_scale_feat, 1, bias=bias))
        modules_body.append(SimpleGate2())
        modules_body.append(CALayer(channels, bias=bias))
        modules_body.append(nn.Conv2d(n_scale_feat, channels, 1, bias=bias))

        self.body = nn.Sequential(*modules_body)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        res = self.body(self.norm(x))
        res = x + res * self.beta
        return res


class CAB2(nn.Module):
    def __init__(self, channels: int, kernel_size, bias=False, add_channels=0) -> None:
        super(CAB2, self).__init__()
        modules_body = []
        scale_factor = 1
        self.n_feat = channels
        self.add_channel = add_channels
        n_scale_feat = int(scale_factor * channels)
        self.conv1 = nn.Conv2d(self.add_channel, self.add_channel, 3, bias=bias, padding=1, groups=self.add_channel)
        self.norm = LayerNorm(self.add_channel + channels)

        modules_body.append(nn.Conv2d(channels + self.add_channel, n_scale_feat * 2, 1, bias=bias))
        modules_body.append(RepConv2(n_scale_feat * 2, bias))
        modules_body.append(SimpleGate())
        modules_body.append(RepConv(n_scale_feat, kernel_size, bias))
        modules_body.append(nn.Conv2d(n_scale_feat, 2 * n_scale_feat, 1, bias=bias))
        modules_body.append(SimpleGate2())
        modules_body.append(CALayer(channels, bias=bias))
        modules_body.append(nn.Conv2d(n_scale_feat, channels, 1, bias=bias))

        self.body = nn.Sequential(*modules_body)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x_input: Tensor) -> Tensor:
        shortcut, hw = x_input[:, 0:self.n_feat], x_input[:, self.n_feat:]
        hw = self.conv1(hw)
        res = self.body(self.norm(torch.cat((shortcut, hw), dim=1)))
        res = shortcut + res * self.beta
        return res


class HWT(nn.Module):
    def __init__(self, channel_in):
        super(HWT, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out[:, :self.channel_in, :, :], out[:, self.channel_in:self.channel_in * 4, :, :]
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)


class MLP(nn.Module):
    """
    多层感知机
    """

    def __init__(self, channels: int) -> None:
        super(MLP, self).__init__()

        hidden_channels = int(channels * 2.66)
        self.project_in = nn.Linear(channels, hidden_channels)
        self.project_out = nn.Linear(hidden_channels, channels)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x: Tensor, f1, f2) -> Tensor:
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> (B T) H W C')
        x = self.project_in(x)
        x = self.act(x)
        x = self.project_out(x)
        x = rearrange(x, '(B T) H W C -> B T C H W', B=B, T=T)

        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = int(dim * 2.66)
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, f1, f2):
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> (B T) H W C')
        x = self.linear1(x)
        x = rearrange(x, '(B T) H W C -> (B T) C H W', B=B, T=T)
        x = self.dwconv(x)
        x = rearrange(x, '(B T) C H W -> (B T) H W C', B=B, T=T)
        x = self.linear2(x)
        x = rearrange(x, '(B T) H W C -> B T C H W', B=B, T=T)
        return x


class SFFN(nn.Module):
    def __init__(self, channels: int) -> None:
        super(SFFN, self).__init__()

        self.conv1 = nn.Conv2d(channels, int(channels * 2.66), kernel_size=1)
        self.conv2 = nn.Conv2d(int(channels * 2.66) // 2, channels, kernel_size=1)

    def forward(self, x: Tensor, f1, f2) -> Tensor:
        B, T, C, H, W = x.shape
        x = rearrange(x, 'B T C H W -> (B T) C H W')
        x = self.conv1(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = x1 * x2
        x = self.conv2(x)
        x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)

        return x


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, f1, f2):
        x = rearrange(x, 'B T C H W -> (B T) C H W')
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2

        x = self.project_out(x)
        x = rearrange(x, '(B T) C H W -> B T C H W', B=1)
        return x


class GDFN(nn.Module):
    """
    3D门控深度卷积前馈网络
    """

    def __init__(self, channels: int) -> None:
        super(GDFN, self).__init__()
        hidden_channels = int(channels * 2.66)
        self.conv_in = nn.Conv3d(channels, hidden_channels * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_out = nn.Conv3d(hidden_channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.dconv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.dconv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)

        self.act = nn.GELU()

        self.flow_conv = nn.Conv2d(4, hidden_channels, kernel_size=1)

    def forward(self, x: Tensor, forward_flow: Tensor, backward_flow: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> B C T H W')
        x = self.conv_in(x)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x1, x2 = torch.chunk(x, 2, dim=1)

        flow = torch.cat((forward_flow, backward_flow), dim=1)
        flow = self.flow_conv(flow)
        x1 = x1 * torch.sigmoid(flow)

        x1 = self.dconv1(x1)
        x1 = self.act(x1)
        x2 = self.dconv2(x2)
        x = x1 * x2

        x = rearrange(x, '(B T) C H W -> B C T H W', B=B, T=T)
        x = self.conv_out(x)
        x = rearrange(x, 'B C T H W -> B T C H W')

        return x


class MambaBlock(nn.Module):
    def __init__(self, channels: int, ffn_type: str = 'GDFN') -> None:
        super(MambaBlock, self).__init__()

        self.norm = LayerNorm(channels, LayerNorm_type='WithBias')
        self.mamba = MMamba(channels)
        if ffn_type == 'GDFN':
            self.ffn = GDFN(channels)
        elif ffn_type == 'MLP':
            self.ffn = MLP(channels)
        elif ffn_type == 'DFFN':
            self.ffn = DFFN(channels)
        elif ffn_type == 'LeFF':
            self.ffn = LeFF(channels)
        elif ffn_type == 'SFFN':
            self.ffn = SFFN(channels)
        elif ffn_type == 'None':
            self.ffn = None

    def forward(self, x: Tensor, forward_flow: Tensor, backward_flow: Tensor, angles: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> (B T) C H W')
        res = x
        x = self.norm(x)
        x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)
        x = self.mamba(x, forward_flow, backward_flow, angles)
        x = x + res

        if self.ffn is not None:
            res = x
            x = rearrange(x, 'B T C H W -> (B T) C H W')
            x = self.norm(x)
            x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)
            x = self.ffn(x, forward_flow, backward_flow)
            x = x + res

        return x


class MSS2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super(MSS2D, self).__init__()

        self.scan_transform = ScanTransform('diagonal', 1, 'add')

        self.conv_in = nn.Conv3d(channels, channels * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_out = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.dconv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)
        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

        self.MSS2D1 = SS2D(channels)
        self.MSS2D2 = SS2D(channels)
        self.out_norm = nn.LayerNorm(channels)

    def scan(self, x: Tensor, angles: Tensor) -> tuple:
        B, C, H, W = x.shape

        mask_0 = (0 <= angles) & (angles < 22.5) | (157.5 < angles) & (angles <= 180)
        mask_22 = (22.5 <= angles) & (angles < 67.5)
        mask_67 = (67.5 <= angles) & (angles < 112.5)
        mask_112 = (112.5 <= angles) & (angles <= 157.5)

        feat1_result = torch.zeros(B, C, H * W, device=x.device, dtype=x.dtype)
        feat2_result = torch.zeros(B, C, H * W, device=x.device, dtype=x.dtype)

        # 水平扫描
        if mask_0.any():
            idx_0 = mask_0.nonzero().squeeze(1)
            feat1_0 = rearrange(x[idx_0], 'B C H W -> B C (H W)')
            feat2_0 = torch.flip(x[idx_0], [-2, -1])
            feat2_0 = rearrange(feat2_0, 'B C H W -> B C (H W)')
            feat1_result[idx_0] = feat1_0
            feat2_result[idx_0] = feat2_0

        # 副对角线扫描
        if mask_22.any():
            idx_22 = mask_22.nonzero().squeeze(1)
            feat1_22 = x[idx_22]
            feat2_22 = torch.flip(x[idx_22], [-2, -1])
            feat1_22 = self.scan_transform.apply_scan_transform(feat1_22.view(-1, 1, C, H * W), (H, W))
            feat2_22 = self.scan_transform.apply_scan_transform(feat2_22.view(-1, 1, C, H * W), (H, W))
            feat1_result[idx_22] = feat1_22.squeeze(1)
            feat2_result[idx_22] = feat2_22.squeeze(1)

        # 情况3: 垂直扫描
        if mask_67.any():
            idx_67 = mask_67.nonzero().squeeze(1)
            feat1_67 = rearrange(x[idx_67], 'B C H W -> B C W H')
            feat1_67 = rearrange(feat1_67, 'B C W H -> B C (W H)')
            feat2_67 = torch.flip(feat1_67, dims=[-1])
            feat1_result[idx_67] = feat1_67
            feat2_result[idx_67] = feat2_67

        # 情况4: 主对角线扫描 (112.5-157.5)
        if mask_112.any():
            # print(4)
            idx_112 = mask_112.nonzero().squeeze(1)
            feat1_112 = torch.flip(x[idx_112], dims=[-1])
            feat2_112 = torch.flip(x[idx_112], [-2, -1])
            feat2_112 = torch.flip(feat2_112, [-1])  # 修正翻转顺序
            feat1_112 = self.scan_transform.apply_scan_transform(feat1_112.view(-1, 1, C, H * W), (H, W))
            feat2_112 = self.scan_transform.apply_scan_transform(feat2_112.view(-1, 1, C, H * W), (H, W))
            feat1_result[idx_112] = feat1_112.squeeze(1)
            feat2_result[idx_112] = feat2_112.squeeze(1)

        return feat1_result, feat2_result

    def scan_reverse(self, x1: Tensor, x2: Tensor, angles: Tensor, size: tuple) -> tuple:
        """
        向量化版本的扫描还原函数

        Args:
            x1: [B, C, L] 扫描后的特征1
            x2: [B, C, L] 扫描后的特征2
            angles: [B] 角度张量
            size: (H, W) 原始空间尺寸

        Returns:
            feat1_list: [B, C, H, W] 还原后的特征1
            feat2_list: [B, C, H, W] 还原后的特征2
        """
        B, C, L = x1.shape
        H, W = size
        assert L == H * W, "展平长度必须等于H*W"

        # 创建角度掩码
        mask_0 = (0 <= angles) & (angles < 22.5) | (157.5 < angles) & (angles <= 180)
        mask_22 = (22.5 <= angles) & (angles < 67.5)
        mask_67 = (67.5 <= angles) & (angles < 112.5)
        mask_112 = (112.5 <= angles) & (angles <= 157.5)

        # 初始化结果张量
        feat1_result = torch.zeros(B, C, H, W, device=x1.device, dtype=x1.dtype)
        feat2_result = torch.zeros(B, C, H, W, device=x1.device, dtype=x1.dtype)

        # 情况1: 水平扫描还原 (0-22.5, 157.5-180)
        if mask_0.any():
            idx_0 = mask_0.nonzero().squeeze(1)
            feat1_0 = rearrange(x1[idx_0], 'B C (H W) -> B C H W', H=H, W=W)
            feat2_0 = rearrange(x2[idx_0], 'B C (H W) -> B C H W', H=H, W=W)
            feat2_0 = torch.flip(feat2_0, dims=[-2, -1])

            feat1_result[idx_0] = feat1_0
            feat2_result[idx_0] = feat2_0

        # 情况2: 副对角线扫描还原 (22.5-67.5)
        if mask_22.any():
            idx_22 = mask_22.nonzero().squeeze(1)
            feat1_22 = self.scan_transform.restore_scan_transform(
                x1[idx_22].unsqueeze(0), (H, W)
            )
            feat2_22 = self.scan_transform.restore_scan_transform(
                x2[idx_22].unsqueeze(0), (H, W)
            )
            feat2_22 = torch.flip(feat2_22, [-2, -1])

            feat1_result[idx_22] = feat1_22.reshape(-1, C, size[0], size[1])
            feat2_result[idx_22] = feat2_22.reshape(-1, C, size[0], size[1])

        # 情况3: 垂直扫描还原 (67.5-112.5)
        if mask_67.any():
            idx_67 = mask_67.nonzero().squeeze(1)
            feat1_67 = rearrange(x1[idx_67], 'B C (W H) -> B C H W', H=H, W=W)
            feat2_67 = rearrange(torch.flip(x2[idx_67], dims=[-1]), 'B C (W H) -> B C H W', H=H, W=W)
            # feat1_67 = feat1_67.mT
            # feat2_67 = torch.flip(feat2_67, dims=[-2, -1]).mT

            feat1_result[idx_67] = feat1_67
            feat2_result[idx_67] = feat2_67

        # 情况4: 主对角线扫描还原 (112.5-157.5)
        if mask_112.any():
            idx_112 = mask_112.nonzero().squeeze(1)
            feat1_112 = self.scan_transform.restore_scan_transform(
                x1[idx_112].unsqueeze(0), (H, W)
            )
            feat2_112 = self.scan_transform.restore_scan_transform(
                x2[idx_112].unsqueeze(0), (H, W)
            )
            feat1_112 = torch.flip(feat1_112, dims=[-1])
            feat2_112 = torch.flip(feat2_112, [-2, -1])
            feat2_112 = torch.flip(feat2_112, [-1])

            feat1_result[idx_112] = feat1_112.reshape(-1, C, size[0], size[1])
            feat2_result[idx_112] = feat2_112.reshape(-1, C, size[0], size[1])

        return feat1_result, feat2_result

    def forward(self, x: Tensor, angels: Tensor) -> None:
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> B C T H W')
        x = self.conv_in(x)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.dconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x11, x12 = self.scan(x, angels)
        x11 = self.MSS2D1(x11)
        x12 = self.MSS2D2(x12)
        x11, x12 = self.scan_reverse(x11, x12, angels, (H, W))
        x1 = x11 + x12
        x1 = self.fusion_conv(x1)

        x1 = self.out_norm(rearrange(x1, 'B C H W -> B (H W) C'))
        x2 = self.act(x2)
        x1 = rearrange(x1, 'B (H W) C -> B C H W', H=H)

        x = x1 * x2
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        x = self.conv_out(x)
        x = rearrange(x, 'B C T H W -> B T C H W')

        return x


class MMamba(nn.Module):
    def __init__(self, channels: int) -> None:
        super(MMamba, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.resconv = ResBlock(channels)
        self.MSS2D = MSS2D(channels)

    def forward(self, x: Tensor, forward_flow: Tensor, backward_flow: Tensor, angels: Tensor) -> Tensor:
        B, T, C, H, W = x.shape
        x_forward = x
        x_backward = x

        x_forward = torch.roll(x_forward, shifts=[1], dims=[1])
        x_backward = torch.roll(x_backward, shifts=[-1], dims=[1])
        x_forward[:, 0, ...] = 0.0
        x_backward[:, -1, ...] = 0.0

        forward_flow = rearrange(forward_flow, 'B C H W -> B H W C')
        backward_flow = rearrange(backward_flow, 'B C H W -> B H W C')
        x_forward = rearrange(x_forward, 'B T C H W -> (B T) C H W')
        x_backward = rearrange(x_backward, 'B T C H W -> (B T) C H W')
        x = rearrange(x, 'B T C H W -> (B T) C H W')

        wrap_forward = flow_warp(x_forward, backward_flow)
        wrap_backward = flow_warp(x_backward, forward_flow)

        wrap_forward = self.conv1(wrap_forward)
        wrap_backward = self.conv2(wrap_backward)
        x = torch.cat((wrap_forward, x, wrap_backward), dim=1)
        x = self.resconv(self.conv3(x))
        x = rearrange(x, '(B T) C H W -> B T C H W', B=B)
        x = self.MSS2D(x, angels)

        return x


class MambaLayer(nn.Module):
    def __init__(self, channels: int, mamba_blocks: list, shift: bool = False, ffn_type: str = 'GDFN') -> None:
        super(MambaLayer, self).__init__()

        add_channels = channels
        self.shift = shift

        self.Encoder = nn.Sequential(
            *[MambaBlock(channels, ffn_type) for _ in range(mamba_blocks[0])]
        )
        self.latent = nn.Sequential(
            *[MambaBlock(channels + add_channels, ffn_type) for _ in range(mamba_blocks[1])]
        )
        self.Decoder = nn.Sequential(
            *[MambaBlock(channels, ffn_type) for _ in range(mamba_blocks[0])]
        )

        if self.shift:
            self.ShiftBlock1 = ShiftBlock(channels)
            self.ShiftBlock2 = ShiftBlock(channels + add_channels)
            self.ShiftBlock3 = ShiftBlock(channels + add_channels)
            self.ShiftBlock4 = ShiftBlock(channels)
        else:
            self.ShiftBlock1 = None
            self.ShiftBlock2 = None
            self.ShiftBlock3 = None
            self.ShiftBlock4 = None

        self.down = DownSampling(in_channels=channels, out_channels=channels + add_channels)
        self.up = UpSampling(in_channels=channels + add_channels, out_channels=channels)
        self.resconv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x: Tensor,
                forward_flow1: Tensor,
                backward_flow1: Tensor,
                forward_flow2: Tensor,
                backward_flow2: Tensor,
                angels: Tensor) -> Tensor:
        B, T, C, H, W = x.shape
        res = x

        if self.shift:
            x = rearrange(x, 'B T C H W -> (B T) C H W')
            x = self.ShiftBlock1(x)
            x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)

        for block in self.Encoder:
            x = block(x, forward_flow1, backward_flow1, angels)

        res1 = x
        x = rearrange(self.down(rearrange(x, 'B T C H W -> (B T) C H W')), '(B T) C H W -> B T C H W', B=B)

        if self.shift:
            x = rearrange(x, 'B T C H W -> (B T) C H W')
            x = self.ShiftBlock2(x)
            x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)

        for block in self.latent:
            x = block(x, forward_flow2, backward_flow2, angels)

        if self.shift:
            x = rearrange(x, 'B T C H W -> (B T) C H W')
            x = self.ShiftBlock3(x)
            x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)

        x = rearrange(x, 'B T C H W -> (B T) C H W')

        x = self.up(x)
        x = rearrange(self.resconv(torch.cat((x, rearrange(res1, 'B T C H W -> (B T) C H W')), dim=1)),
                      '(B T) C H W -> B T C H W', B=B)

        for block in self.Decoder:
            x = block(x, forward_flow1, backward_flow1, angels)
        if self.shift:
            x = rearrange(x, 'B T C H W -> (B T) C H W')
            x = self.ShiftBlock4(x)
            x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)

        return x + res


class GatePropagation(nn.Module):
    def __init__(self, channels: int) -> None:
        super(GatePropagation, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = x1 * torch.sigmoid(self.conv1(x1))
        x2 = x2 * torch.sigmoid(self.conv1(x2))

        return x1 + x2


class MultiScaleFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super(MultiScaleFusion, self).__init__()

        self.conv_in = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.conv_out = nn.Conv2d(channels * 3, channels, kernel_size=1)

        self.DP1 = IDynamicDWConv(channels, kernel_size=3, group_channels=1, down=4, conv_group=1, resblocks=1)
        self.DP2 = IDynamicDWConv(channels, kernel_size=5, group_channels=1, down=4, conv_group=1, resblocks=1)
        self.DP3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels)
        )

        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        w = self.atten(x)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        x1 = self.DP1(x1)
        x2 = self.DP2(x2)
        x3 = self.DP3(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x * w
        x = self.conv_out(x)

        return x


class AMDFF(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super(AMDFF, self).__init__()

        self.conv1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=1)

        self.gate = GatePropagation(channels)
        self.DP = MultiScaleFusion(channels)

        self.reslayer = nn.Sequential(
            *[ResBlock(channels) for _ in range(blocks)]
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.gate(x)
        x = self.DP(x)
        x = self.reslayer(x)

        return x


class EDTFF(nn.Module):
    def __init__(self, channels: int, blocks: int, fusion_ratio: float) -> None:
        super(EDTFF, self).__init__()

        dp_channels = int(channels * fusion_ratio)
        self.gate = GatePropagation(dp_channels)

        self.conv1 = nn.Conv2d(channels * 2, dp_channels * 2, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.1)

        self.DP1 = IDynamicDWConv(dp_channels, kernel_size=3, group_channels=1, down=4, conv_group=1, resblocks=1)

        self.reslayer = nn.Sequential(
            *[ResBlock(channels) for _ in range(blocks)]
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.gate(x)
        x = self.DP1(x)
        x = self.reslayer(x)

        return x


class EDTFFL(nn.Module):
    def __init__(self, channels: int, blocks: int, fusion_ratio: float) -> None:
        super(EDTFFL, self).__init__()

        dp_channels = int(channels * fusion_ratio)

        self.conv1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.act = nn.LeakyReLU(negative_slope=0.1)

        self.DP1 = IDynamicDWConv(dp_channels, kernel_size=3, group_channels=1, down=4, conv_group=1, resblocks=1)
        self.DP2 = IDynamicDWConv(dp_channels, kernel_size=5, group_channels=1, down=4, conv_group=1, resblocks=1)

        self.conv3 = nn.Conv2d(dp_channels, channels, kernel_size=1)

        self.reslayer = nn.Sequential(
            *[ResBlock(channels) for _ in range(blocks)]
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.act(x)

        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = x1 * torch.sigmoid(self.avg(x1))
        x2 = x2 * torch.sigmoid(self.avg(x2))
        x = self.conv2(x1 + x2)

        x = self.DP1(x) + self.DP2(x)
        x = self.conv3(x)
        x = self.reslayer(x)

        return x


class ForwardPropagation(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super(ForwardPropagation, self).__init__()

        self.EDTFF = EDTFF(channels, blocks, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        feat_list = []
        feat = x.new_zeros((B, C, H, W), requires_grad=True)
        for t in range(0, T):
            x1 = x[:, t, ...]
            out = self.EDTFF(x1, feat)
            feat_list.append(out)

        feat = torch.stack(feat_list, dim=1)
        return feat


class BackwardPropagation(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super(BackwardPropagation, self).__init__()

        self.EDTFF = EDTFF(channels, blocks, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        feat_list = []
        feat = x.new_zeros((B, C, H, W), requires_grad=True)
        for t in range(T - 1, -1, -1):
            x1 = x[:, t, ...]
            out = self.EDTFF(x1, feat)
            feat_list.append(out)
        feat_list.reverse()

        feat = torch.stack(feat_list, dim=1)
        return feat


class ForwardPropagationWithResBlock(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super(ForwardPropagationWithResBlock, self).__init__()

        self.Fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            *[CAB(channels, 1) for _ in range(15)]
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        feat_list = []
        feat = x.new_zeros((B, C, H, W), requires_grad=True)
        for t in range(0, T):
            x1 = x[:, t, ...]
            out = torch.cat((x1, feat), dim=1)
            out = self.Fusion(out)
            feat_list.append(out)

        feat = torch.stack(feat_list, dim=1)
        return feat


class BackwardPropagationWithResBlock(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super(BackwardPropagationWithResBlock, self).__init__()

        self.Fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            *[CAB(channels, 1) for _ in range(15)]
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        feat_list = []
        feat = x.new_zeros((B, C, H, W), requires_grad=True)
        for t in range(T - 1, -1, -1):
            x1 = x[:, t, ...]
            out = torch.cat((x1, feat), dim=1)
            out = self.Fusion(out)
            feat_list.append(out)
        feat_list.reverse()

        feat = torch.stack(feat_list, dim=1)
        return feat


class ForwardPropagationL(nn.Module):
    def __init__(self, channels: int, blocks: int, fusion_ratio: float) -> None:
        super(ForwardPropagationL, self).__init__()

        self.EDTFF = EDTFFL(channels, blocks, fusion_ratio)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        feat_list = []
        feat = x.new_zeros((B, C, H, W), requires_grad=True)
        for t in range(0, T):
            x1 = x[:, t, ...]
            out = self.EDTFF(x1, feat)
            feat_list.append(out)

        feat = torch.stack(feat_list, dim=1)
        return feat


class BackwardPropagationL(nn.Module):
    def __init__(self, channels: int, blocks: int, fusion_ratio: float) -> None:
        super(BackwardPropagationL, self).__init__()

        self.EDTFF = EDTFFL(channels, blocks, fusion_ratio)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        feat_list = []
        feat = x.new_zeros((B, C, H, W), requires_grad=True)
        for t in range(T - 1, -1, -1):
            x1 = x[:, t, ...]
            out = self.EDTFF(x1, feat)
            feat_list.append(out)
        feat_list.reverse()

        feat = torch.stack(feat_list, dim=1)
        return feat


class UNet(nn.Module):
    """
    2D UNet
    """

    def __init__(self, channels: int, reduction: int, block) -> None:
        super(UNet, self).__init__()
        add_channels = 4

        self.encoder_level1 = [block(channels, reduction) for _ in range(1)]
        self.encoder_level2 = [block(channels + add_channels, reduction) for _ in range(1)]
        self.encoder_level3 = [block(channels + 2 * add_channels, reduction) for _ in range(1)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.down12 = DownSample(channels, add_channels)
        self.down23 = DownSample(channels + add_channels, add_channels)

        self.decoder_level1 = [block(channels, reduction) for _ in range(1)]
        self.decoder_level2 = [block(channels + add_channels, reduction) for _ in range(1)]
        self.decoder_level3 = [block(channels + add_channels * 2, reduction) for _ in range(1)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.skip_attn2 = nn.Conv2d(channels + add_channels, channels + add_channels, kernel_size=3, padding=1)
        self.up21 = SkipUpSample(channels, add_channels)
        self.up32 = SkipUpSample(channels + add_channels, add_channels)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)

        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return dec1


class ShiftBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ShiftBlock, self).__init__()
        self.number = channels // 2 // 8

        self.encoder_level1 = [CAB2(channels, kernel_size=5, add_channels=8 * self.number),
                               CAB1(channels, kernel_size=5)]
        self.encoder_level1_1 = [CAB2(channels, kernel_size=5, add_channels=8 * self.number),
                                 CAB1(channels, kernel_size=5)]
        self.encoder_level1_2 = [CAB2(channels, kernel_size=5, add_channels=8 * self.number),
                                 CAB1(channels, kernel_size=5)]
        self.encoder_level1_3 = [CAB2(channels, kernel_size=5, add_channels=8 * self.number),
                                 CAB1(channels, kernel_size=5)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level1_1 = nn.Sequential(*self.encoder_level1_1)
        self.encoder_level1_2 = nn.Sequential(*self.encoder_level1_2)
        self.encoder_level1_3 = nn.Sequential(*self.encoder_level1_3)

    def spatial_shift(self, hw):
        n2 = (self.number - 1) // 2
        n1 = self.number - 2 * n2
        s = 4
        _, _, H, W = hw.shape
        s_out = torch.zeros_like(hw)
        s_out[:, 0 * n2:1 * n2, s * 2:, s * 2:] = hw[:, 0 * n2:1 * n2, :-s * 2, :-s * 2]
        s_out[:, 1 * n2:2 * n2, s * 2:, s:] = hw[:, 1 * n2:2 * n2, :-s * 2, :-s]
        s_out[:, 2 * n2:3 * n2, s * 2:, 0:] = hw[:, 2 * n2:3 * n2, :-s * 2, :]
        s_out[:, 3 * n2:4 * n2, s * 2:, 0:-s] = hw[:, 3 * n2:4 * n2, :-s * 2, s:]
        s_out[:, 4 * n2:5 * n2, s * 2:, 0:-s * 2] = hw[:, 4 * n2:5 * n2, :-s * 2, s * 2:]

        s_out[:, 5 * n2:6 * n2, 0:-s * 2, s * 2:] = hw[:, 5 * n2:6 * n2, s * 2:, :-s * 2]
        s_out[:, 6 * n2:7 * n2, 0:-s * 2, s:] = hw[:, 6 * n2:7 * n2, s * 2:, :-s]
        s_out[:, 7 * n2:8 * n2, 0:-s * 2, 0:] = hw[:, 7 * n2:8 * n2, s * 2:, :]
        s_out[:, 8 * n2:9 * n2, 0:-s * 2, 0:-s] = hw[:, 8 * n2:9 * n2, s * 2:, s:]
        s_out[:, 9 * n2:10 * n2, 0:-s * 2, 0:-s * 2] = hw[:, 9 * n2:10 * n2, s * 2:, s * 2:]

        s_out[:, 10 * n2:11 * n2, s:, s * 2:] = hw[:, 10 * n2:11 * n2, :-s, :-s * 2]
        s_out[:, 11 * n2:12 * n2, s:, 0:-s * 2] = hw[:, 11 * n2:12 * n2, :-s, s * 2:]
        s_out[:, 12 * n2:13 * n2, :, s * 2:] = hw[:, 12 * n2:13 * n2, :, :-s * 2]
        s_out[:, 13 * n2:14 * n2, :, 0:-s * 2] = hw[:, 13 * n2:14 * n2, :, s * 2:]
        s_out[:, 14 * n2:15 * n2, 0:-s, s * 2:] = hw[:, 14 * n2:15 * n2, s:, :-s * 2]
        s_out[:, 15 * n2:16 * n2, 0:-s, 0:-s * 2] = hw[:, 15 * n2:16 * n2, s:, s * 2:]
        s_out[:, 16 * n2 + 0 * n1:16 * n2 + 1 * n1, s:, s:] = hw[:, 16 * n2 + 0 * n1:16 * n2 + 1 * n1, :-s, :-s]
        s_out[:, 16 * n2 + 1 * n1:16 * n2 + 2 * n1, s:, 0:] = hw[:, 16 * n2 + 1 * n1:16 * n2 + 2 * n1, :-s, :]
        s_out[:, 16 * n2 + 2 * n1:16 * n2 + 3 * n1, s:, 0:-s] = hw[:, 16 * n2 + 2 * n1:16 * n2 + 3 * n1, :-s, s:]
        s_out[:, 16 * n2 + 3 * n1:16 * n2 + 4 * n1, :, s:] = hw[:, 16 * n2 + 3 * n1:16 * n2 + 4 * n1, :, :-s]
        s_out[:, 16 * n2 + 4 * n1:16 * n2 + 5 * n1, :, 0:-s] = hw[:, 16 * n2 + 4 * n1:16 * n2 + 5 * n1, :, s:]
        s_out[:, 16 * n2 + 5 * n1:16 * n2 + 6 * n1, 0:-s, s:] = hw[:, 16 * n2 + 5 * n1:16 * n2 + 6 * n1, s:, :-s]
        s_out[:, 16 * n2 + 6 * n1:16 * n2 + 7 * n1, 0:-s, 0:] = hw[:, 16 * n2 + 6 * n1:16 * n2 + 7 * n1, s:, :]
        s_out[:, 16 * n2 + 7 * n1:16 * n2 + 8 * n1, 0:-s, 0:-s] = hw[:, 16 * n2 + 7 * n1:16 * n2 + 8 * n1, s:, s:]
        return s_out

    def channel_shift(self, x: Tensor, div=2, reverse=False) -> Tensor:
        B, C, H, W = x.shape
        slice_c = C // div
        if reverse:
            slice_c = -slice_c
        y = x.reshape(1, B * C, H, W)
        y = torch.roll(y, slice_c, 1).reshape(B, C, H, W)

        if not reverse:
            hw = y[:, 0:8 * self.number, ...]
        else:
            hw = y[:, -8 * self.number:, ...]
        hw = self.spatial_shift(hw)

        return torch.cat((y, hw), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_shift(x)
        x = self.encoder_level1(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_1(x)
        x = self.channel_shift(x)
        x = self.encoder_level1_2(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_3(x)

        return x
