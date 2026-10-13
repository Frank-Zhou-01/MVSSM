import torch
from einops import rearrange
from torch import Tensor
from torch import nn

from basicsr.archs.components import HWT, MambaLayer, BackwardPropagation, ForwardPropagation, \
    ForwardPropagationWithResBlock, BackwardPropagationWithResBlock
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register
class MVSSM_arch(nn.Module):
    """
    Motion-aware Visual State Space Model
    """
    def __init__(self, channels: int, mamba_blocks: list, fusion_blocks: int) -> None:
        super().__init__()

        self.conv_in = nn.Conv3d(3, channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.conv_out = nn.Conv3d(channels, 3, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.FlowAvg = nn.AdaptiveAvgPool2d((1, 1))

        self.WT = HWT(channels)
        self.conv_scale1 = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, groups=3),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, groups=3)
        )
        self.conv_scale2 = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, groups=3),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, groups=3)
        )

        self.MambaLayer = MambaLayer(channels, mamba_blocks)
        self.Backward = ForwardPropagation(channels, fusion_blocks)
        self.Forward = BackwardPropagation(channels, fusion_blocks)

    def forward(self, x: Tensor, forward_flow1: Tensor, backward_flow1: Tensor, forward_flow2: Tensor, backward_flow2: Tensor) -> Tensor:
        res0 = x

        angles = self.FlowAvg(forward_flow1).squeeze(-1).squeeze(-1).detach()
        angles = torch.atan2(angles[:, 1], angles[:, 0])
        angles = (angles * 180) / torch.pi
        angles[angles < 0.0] += 360.0
        angles = angles

        B, T, C, H, W = x.shape
        x = rearrange(x, 'B T C H W -> B C T H W')
        x = self.conv_in(x)
        x = rearrange(x, 'B C T H W -> (B T) C H W')

        low1, high1 = self.WT(x)
        low2, high2 = self.WT(low1)
        high2 = self.conv_scale2(high2)
        low2 = rearrange(low2, '(B T) C H W -> B T C H W', B=B, T=T)

        low2 = rearrange(self.MambaLayer(low2, forward_flow1, backward_flow1, forward_flow2, backward_flow2, angles), 'B T C H W -> (B T) C H W')

        low1 = torch.cat((low2, high2), dim=1)
        low1 = self.WT(low1, rev=True)

        low1 = rearrange(low1, '(B T) C H W -> B T C H W', B=B, T=T)
        low1 = self.Backward(low1)
        low1 = self.Forward(low1)
        low1 = rearrange(low1, 'B T C H W -> (B T) C H W')

        high1 = self.conv_scale1(high1)
        low1 = torch.cat((low1, high1), dim=1)
        x = self.WT(low1, rev=True)

        x = rearrange(x, '(B T) C H W -> B C T H W', B=B, T=T)
        x = self.conv_out(x)
        x = rearrange(x, 'B C T H W -> B T C H W')

        return x.contiguous() + res0
