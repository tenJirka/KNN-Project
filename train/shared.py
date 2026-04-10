"""Classses which can be used in different files"""

import pytorch_lightning as pl
import timm
import torch
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


class ReIDLightningModel(pl.LightningModule):
    def __init__(self, model, criterion_metric, miner):
        super().__init__()
        self.model = model
        self.criterion_metric = criterion_metric
        self.miner = miner

    def training_step(self, batch, batch_idx):
        images, labels = batch

        features = self.model(images)

        # Use mainer to get hardest pairs from batch
        hard_pairs = self.miner(features, labels)

        # Calculate loss only for hard pairs
        loss = self.criterion_metric(features, labels, hard_pairs)

        self.log("train_loss", loss, prog_bar=True)

        # NOTE: Logging how many hard pairs was found
        self.log("mined_pairs", float(len(hard_pairs[0])), prog_bar=False)

        return loss

    def configure_optimizers(self):
        # Use same optimizer for loss functions and model itself
        # TODO: Do more research on optimisers and learning rates
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters(), "lr": 0.0003},
                {"params": self.criterion_metric.parameters(), "lr": 0.001},
            ]
        )
        return optimizer
