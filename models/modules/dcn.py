# -*- coding: utf-8 -*-
"""
Deformable Convolution v2 (DCNv2) implementation using torchvision.ops.
Falls back to standard convolution if torchvision.ops is not available.
"""

import torch
from torch import nn

try:
    from torchvision.ops import DeformConv2d
    HAS_DCN = True
except ImportError:
    HAS_DCN = False


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 wrapper.
    Learns offset and mask for each convolution position.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if not HAS_DCN:
            # Fallback to standard convolution
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation,
                                  groups=groups, bias=bias)
            self.use_dcn = False
            return

        self.use_dcn = True
        # offset: 2 * kernel_size^2 channels (x,y offsets per position)
        # mask: kernel_size^2 channels (modulation scalar per position)
        offset_channels = 2 * kernel_size * kernel_size
        mask_channels = kernel_size * kernel_size

        self.offset_conv = nn.Conv2d(in_channels, offset_channels, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation,
                                      bias=True)
        self.mask_conv = nn.Conv2d(in_channels, mask_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation,
                                    bias=True)
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)

        # Initialize offsets to zero and masks to ones
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)

    def forward(self, x):
        if not self.use_dcn:
            return self.conv(x)

        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.dcn(x, offset, mask)
