import torch.nn.functional as F
import torch
import torch.nn as nn


class double_conv(nn.Module):
    '''
    (conv => ReLU) * 2
    Most U-net models use batch normalization, but I
    found it caused discontinuities. Maybe worth
    looking into batch normalization better.
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    '''
    Downsampling operation with maxpooling followed by
    convolution operations. Trainable downsampling methods
    may be worth exploring.
    '''
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.mp(x)
        x = self.conv(x)
        return x


class up(nn.Module):
    '''
    Upsampling operation. I tried upsampling with bilinear upsampling
    and it worked just as well as transpose convolution.
    May be better to stick with this than use transpose convolution.
    '''
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                   diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class uptranspose(nn.Module):
    '''
    This is using transpose convolution where upsampling is learnt.
    This uses more memory and results aren't very different from
    bilinear upsampling.
    '''

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    '''
    This is the model, designed experimentally.

    '''
    def __init__(self, n_channels, n_classes=1):
        super(UNet, self).__init__()
        self.inc = double_conv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = nn.Conv2d(64, n_classes,1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        #return F.sigmoid(x) --Can use final activation
        return x
