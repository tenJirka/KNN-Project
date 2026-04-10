"""Classses which can be used in different files"""

import timm
import torch.nn as nn


class GenericReIDModel(nn.Module):
    """Helper class for generic model abstarction (Resnet, ViT,...)"""

    def __init__(self, model_name, pretrained=True):
        super().__init__()

        # Remove model cassifier
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        self.feature_dim = self.backbone.num_features

    def forward(self, x):
        # Return only features
        return self.backbone(x)
