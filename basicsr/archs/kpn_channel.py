import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DynamicDWConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups

        self.tokernel = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, kernel_size ** 2 * self.channels, kernel_size=1, stride=1)
        )

    def forward(self, x: Tensor, k: Tensor) -> Tensor:
        B, C, H, W = x.shape

        k = self.tokernel(k)
        k = k.view(B * self.channels, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, H, W), k, self.bias.repeat(B), stride=self.stride, padding=self.padding, groups=B * self.groups)
        x = x.view(B, C, x.shape[-2], x.shape[-1])

        return x
