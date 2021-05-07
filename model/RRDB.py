import torch
from torch import nn
import torch.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channel=64, growth_channel=32, scale_ratio: float = 0.2):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, growth_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel+growth_channel, growth_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel+2*growth_channel, growth_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel+3*growth_channel, growth_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=.2, inplace=True)
        )

        self.conv5 = nn.Conv2d(in_channel+4*growth_channel, in_channel, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_cat = torch.cat([x, conv1], dim=1)

        conv2 = self.conv2(conv1_cat)
        conv2_cat = torch.cat([x, conv1, conv2], dim=1)

        conv3 = self.conv3(conv2_cat)
        conv3_cat = torch.cat([x, conv1, conv2, conv3], dim=1)

        conv4 = self.conv4(conv3_cat)
        conv4_cat = torch.cat([x, conv1, conv2, conv3, conv4], dim=1)

        pred = self.conv5(conv4_cat)

        return pred * self.scale_ratio + x


class RRDB(nn.Module):
    def __init__(self, in_channel=64, growth_channel=32, scale_ratio:float = 0.2):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(in_channel, growth_channel, scale_ratio)
        self.RDB2 = ResidualDenseBlock(in_channel, growth_channel, scale_ratio)
        self.RDB3 = ResidualDenseBlock(in_channel, growth_channel, scale_ratio)

    def forward(self, x):
        pred = self.RDB1(x)
        pred = self.RDB2(pred)
        pred = self.RDB3(pred)

        return pred * 0.2 + x

