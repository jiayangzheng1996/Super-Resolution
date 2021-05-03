import math
import torch
from torch import nn
import torch.nn.functional as F
from RRDB import RRDB

class Generator(nn.Module):
    def __init__(self, num_rrdb_blocks=16, scaling_factor=8):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        blocks = []
        for _ in range(num_rrdb_blocks):
            rrdb = RRDB(in_channel=64, growth_channel=32, scale_ratio=0.2)
            blocks.append(rrdb)
        self.blocks = nn.Sequential(*blocks)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.scale_loop = int(math.log(scaling_factor, 2))

        up = []
        for _ in range(self.scale_loop):
            conv_up = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=.2, inplace=True)
            )
            up.append(conv_up)
        self.up = nn.ModuleList(up)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=.2, inplace=True)
        )

        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, lr_img):
        conv1 = self.conv1(lr_img)
        block_out = self.blocks(conv1)
        conv2 = self.conv2(block_out)
        up_out = conv2
        for i in range(self.scale_loop):
            upsample = F.interpolate(up_out, scale_factor=2, mode="nearest")
            up_out = self.up[i](upsample)
        conv3 = self.conv3(up_out)
        pred = self.conv4(conv3)

        return (torch.tanh(pred) + 1) / 2



def conv_leakyrelu(in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):
    if batchnorm:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(negative_slope=.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=.2, inplace=True)
        )


class Discriminator(nn.Module):
    def __init__(self, image_size=512):
        super(Discriminator, self).__init__()

        feature_size = image_size // 32

        conv1 = conv_leakyrelu(3, 64, 3, 1, 1)
        conv2 = conv_leakyrelu(64, 64, 3, 2, 1, batchnorm=True)
        conv3 = conv_leakyrelu(64, 128, 3, 1, 1, batchnorm=True)
        conv4 = conv_leakyrelu(128, 128, 3, 2, 1, batchnorm=True)
        conv5 = conv_leakyrelu(128, 256, 3, 1, 1, batchnorm=True)
        conv6 = conv_leakyrelu(256, 256, 3, 2, 1, batchnorm=True)
        conv7 = conv_leakyrelu(256, 512, 3, 1, 1, batchnorm=True)
        conv8 = conv_leakyrelu(512, 512, 3, 2, 1, batchnorm=True)
        conv9 = conv_leakyrelu(512, 512, 3, 1, 1, batchnorm=True)
        conv10 = conv_leakyrelu(512, 512, 3, 2, 1, batchnorm=True)
        conv_layers = [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10]
        self.conv_layers = nn.Sequential(*conv_layers)

        self.classifier = nn.Sequential(
            nn.Linear(512*feature_size*feature_size, 128),
            nn.LeakyReLU(negative_slope=.2, inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, img):
        conv_out = self.conv_layers(img)
        pred = self.classifier(conv_out.flatten(start_dim=1))
        return pred

