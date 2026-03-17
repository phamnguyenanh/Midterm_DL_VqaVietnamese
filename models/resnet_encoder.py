import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50Encoder(nn.Module):
    """ResNet50 encoder with global feature and spatial regions."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self):
        for p in self.layer4.parameters():
            p.requires_grad = True

    def forward(self, images: torch.Tensor):
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        global_feat = self.avgpool(x).flatten(1)

        feature_map = F.interpolate(x, size=(14, 14), mode="bilinear", align_corners=False)
        regions = feature_map.flatten(2).permute(0, 2, 1)
        return global_feat, regions
