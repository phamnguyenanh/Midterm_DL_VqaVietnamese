import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Simple CNN encoder returning global feature and spatial regions."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, out_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images: torch.Tensor):
        x = F.relu(self.conv1(images))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))

        feature_map = self.adaptive_pool(x)
        global_feat = self.gap(x).flatten(1)

        regions = feature_map.flatten(2).permute(0, 2, 1)
        return global_feat, regions
