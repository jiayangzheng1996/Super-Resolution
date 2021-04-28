import torch
import scipy.ndimage
from torch import nn
import torchvision
from torchvision import models
from torch import functional as F
from torchvision.models._utils import IntermediateLayerGetter


def conv3x3_batchnorm(in_channel, out_channel, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channel)
    )


def stack(img, n, sigma=1):
    G, L = [img], []
    n_sigma = sigma

    for i in range(n):
        gaussian_img = scipy.ndimage.gaussian_filter(img, sigma=n_sigma)
        G.append(gaussian_img)
        n_sigma *= 2

    for i in range(len(G)-1):
        laplacian_img = G[i] - G[i+1]
        laplacian_img = torch.tensor(laplacian_img).cuda()
        L.append(laplacian_img)

    return G, L


class RefinementNet(nn.Module):
    def __init__(self, pyramid_level=5, num_channels=3):
        super(RefinementNet, self).__init__()
        self.pyramid_level = pyramid_level
        conv_layers = []
        for i in range(pyramid_level):
            conv = conv3x3_batchnorm(3, 64)
            conv_layers.append(conv)
        self.conv_layers = nn.ModuleList(conv_layers)

        self.final_conv = nn.Conv2d(self.pyramid_level*64, num_channels, kernel_size=1)


    def forward(self, gen_image):
        x = gen_image.numpy()
        _, L = stack(x, self.pyramid_level)
        assert len(L) == self.pyramid_level

        outputs = []
        for i in range(len(L)):
            output = self.conv_layers[i](L[i])
            outputs.append(output)
        output = torch.cat(outputs, dim=1)
        pred = self.final_conv(output)
        return pred



if __name__ == '__main__':
    gen_image = torch.ones((4, 3, 512, 512))
    model = RefinementNet().cuda()
    out = model(gen_image)