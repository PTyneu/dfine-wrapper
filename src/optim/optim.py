"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register

__all__ = ["AdamW", "SGD", "SGDIgnoreBetas", "Adam", "MultiStepLR", "CosineAnnealingLR", "OneCycleLR", "LambdaLR"]


SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)


@register()
class SGDIgnoreBetas(optim.SGD):
    """SGD that tolerates the AdamW-only `betas` key inherited from the included base configs."""

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=False, betas=None):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
