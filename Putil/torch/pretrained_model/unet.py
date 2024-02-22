# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F

import Putil.base.logger as plog
UnetLogger = plog.PutilLogConfig('Unet').logger()
UnetLogger.setLevel(plog.DEBUG)
UnetModuleLogger = UnetLogger.getChild('UnetModule')
UnetModuleLogger.setLevel(plog.DEBUG)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, downsample_rate, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

    def forward(self, x):
        x1 = self.inc(x)
        UnetModuleLogger.debug('x1 min: {}; max: {}'.format(x1.min(), x1.max()))
        x2 = self.down1(x1)
        UnetModuleLogger.debug('x2 min: {}; max: {}'.format(x2.min(), x2.max()))
        x3 = self.down2(x2)
        UnetModuleLogger.debug('x3 min: {}; max: {}'.format(x3.min(), x3.max()))
        x4 = self.down3(x3)
        UnetModuleLogger.debug('x4 min: {}; max: {}'.format(x4.min(), x4.max()))
        x5 = self.down4(x4)
        UnetModuleLogger.debug('x5 min: {}; max: {}'.format(x5.min(), x5.max()))
        x = self.up1(x5, x4)
        UnetModuleLogger.debug('x min: {}; max: {}'.format(x.min(), x.max()))
        x = self.up2(x, x3)
        UnetModuleLogger.debug('x min: {}; max: {}'.format(x.min(), x.max()))
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    @property
    def final_out_channels(self):
        return 256 
    pass


class NonLocalAttentionUNet(UNet):
    def __init__(self, downsample_rate, features, input_shape, bilinear=False):
        self._n_channels = input_shape[1] 
        self._bilinear = bilinear
        super(NonLocalAttentionUNet, self).__init__(self._n_channels, downsample_rate, bilinear)

    def forward(self, x):
        x1 = self.inc(x)
        UnetModuleLogger.debug('x1 min: {}; max: {}'.format(x1.min(), x1.max()))
        x2 = self.down1(x1)
        UnetModuleLogger.debug('x2 min: {}; max: {}'.format(x2.min(), x2.max()))
        x3 = self.down2(x2)
        UnetModuleLogger.debug('x3 min: {}; max: {}'.format(x3.min(), x3.max()))
        x4 = self.down3(x3)
        UnetModuleLogger.debug('x4 min: {}; max: {}'.format(x4.min(), x4.max()))
        x5 = self.down4(x4)
        UnetModuleLogger.debug('x5 min: {}; max: {}'.format(x5.min(), x5.max()))
        x = self.up1(x5, x4)
        UnetModuleLogger.debug('x min: {}; max: {}'.format(x.min(), x.max()))
        x = self.up2(x, x3)
        UnetModuleLogger.debug('x min: {}; max: {}'.format(x.min(), x.max()))
        x = self.up3(x, x2)
        UnetModuleLogger.debug('x min: {}; max: {}'.format(x.min(), x.max()))
        x = self.up4(x, x1)
        return x
    pass