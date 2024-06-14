import torch
from torch import nn


def get_heatmap_cond_module(cond_module_args, in_feature_channels):
    module_type = cond_module_args["type"]
    if module_type == "se_block":
        return SqueezeHeatmapCondition(in_feature_channels)
    if module_type == "1x1":
        return Conv2dHeatmapCondition(in_feature_channels)


class Conv2dHeatmapCondition(nn.Module):
    def __init__(self, in_feature_channels):
        super().__init__()
        self.in_feature_channels = in_feature_channels
        self.conv2d = nn.Conv2d(in_feature_channels, in_feature_channels, kernel_size=5, padding="same")
        self.activ_f = nn.LeakyReLU(0.01)

    def forward(self, features, heatmap, with_res=True):
        x = torch.cat([features, heatmap], axis=1)
        x = self.conv2d(x)
        if with_res:
            return self.activ_f(x + features)
        else:
            return self.activ_f(x)


class FatSqueezeHeatmapCondition(nn.Module):
    def __init__(self, in_feature_channels):
        super().__init__()
        self.in_feature_channels = in_feature_channels
        self.se_block = SEBlock(in_feature_channels, r=1)

    def forward(self, x):
        pass


class SqueezeHeatmapCondition(nn.Module):
    def __init__(self, in_feature_channels):
        super().__init__()
        self.in_feature_channels = in_feature_channels
        self.conv2d = Conv2dHeatmapCondition(in_feature_channels)
        self.se_block = SEBlock(in_feature_channels, r=1)

    def forward(self, features, heatmap):
        x = self.conv2d(features, heatmap, with_res=False)
        return features + self.se_block(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.in_channels = in_channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
