import torch
import torch.nn as nn
import torch.nn.functional as F


def get_upsample_l(kind):
    if kind == "bilinear":
        return BilinearUpsample
    elif kind == "nn2d":
        return NNConvUpsample2D
    elif kind == "nn3d":
        return NNConvUpsample3D
    elif kind == "bicubic":
        return BicubicUpsample
    else:
        raise NotImplementedError(f"Upsample layer {kind} not implemented")


class NNConvUpsample2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=4, padding="same")
        self.conv2 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=4, padding="same")

    def forward(self, x, size):
        x = F.interpolate(x, size, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class NNConvUpsample3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.convd2 = nn.Conv3d(self.in_channels, self.in_channels, kernel_size=(1, 3, 3), padding="same")

    def forward(self, x, size):
        x = F.interpolate(x, size, mode="nearest")
        x = self.convd2(x)
        return x


class BilinearUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=True)


class BicubicUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x, size):
        return F.interpolate(x, size, mode="bicubic", align_corners=True)
