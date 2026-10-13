import sys

sys.path.append('/root/autodl-tmp/MVSSM/')

from basicsr.archs.MVSSM_arch import MVSSM_arch
from basicsr.archs.MVSSML_arch import MVSSML_arch

import torch
from thop import profile


def get_params(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("总参数数量和: " + f'{(k / (1024 * 1024)):.2f}M')


def count_flops(name, model):
    device = torch.device('cuda')
    model = model.to(device)
    x = torch.randn((1, 3, 3, 256, 256)).to(device)
    f1 = torch.randn((3, 2, 64, 64)).to(device)
    b1 = torch.randn((3, 2, 64, 64)).to(device)
    f2 = torch.randn((3, 2, 32, 32)).to(device)
    b2 = torch.randn((3, 2, 32, 32)).to(device)

    flops, params = profile(model, inputs=(x, f1, b1, f2, b2))

    print(f'{name}: FLOPS: {flops / (1024 * 1024 * 1024 * 3)}G, Params: {params / (1024 * 1024)}M')


net = MVSSM_arch(channels=64, mamba_blocks=[3, 4], fusion_blocks=15)
netL = MVSSML_arch(channels=64, mamba_blocks=[3, 4], fusion_blocks=15)
count_flops('Net:', net)
count_flops('NetL:', netL)

