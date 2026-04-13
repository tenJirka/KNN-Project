"""Classses which can be used in different files"""

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from util import compute_reid_metrics


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

        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        images, labels, cam_ids = batch

        features = self.model(images)

        # Use mainer to get hardest pairs from batch
        hard_pairs = self.miner(features, labels)

        # Calculate loss only for hard pairs
        loss = self.criterion_metric(features, labels, hard_pairs)

        self.log("train_loss", loss, prog_bar=True)

        # NOTE: Logging how many hard pairs was found
        self.log("mined_pairs", float(len(hard_pairs[0])), prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # batch should now return: images, labels (v_ids), and cam_ids
        images, labels, cam_ids = batch

        # Extract features (normalized for cosine similarity)
        features = self.model(images)
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        # Store results to compute metrics at the end of the epoch
        self.validation_step_outputs.append(
            {
                "features": features.cpu(),
                "labels": labels.cpu(),
                "cam_ids": cam_ids.cpu(),
            }
        )

    def on_validation_epoch_end(self):
        # Concatenate all features, labels, and cam_ids from validation steps
        features = torch.cat(
            [x["features"] for x in self.validation_step_outputs], dim=0
        )
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs], dim=0)
        cam_ids = torch.cat([x["cam_ids"] for x in self.validation_step_outputs], dim=0)

        self.validation_step_outputs.clear()

        # Compute metrics (mAP, CMC) using the extracted features, labels, and cam_ids
        mAP, cmc = compute_reid_metrics(
            features, labels, cam_ids, features, labels, cam_ids, max_rank=1
        )

        rank1 = cmc[0] if len(cmc) > 0 else 0.0

        self.log("val_mAP", mAP, prog_bar=True)
        self.log("val_rank1", rank1, prog_bar=True)

    def configure_optimizers(self):
        # Select LR and optimizer
        # TODO: More reasearch on this
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": 0.00005},
                {"params": self.criterion_metric.parameters(), "lr": 0.0001},
            ],
            weight_decay=0.01,
        )

        warmup_epochs = 10
        total_epochs = 100  # Have to by same as in ./train.py

        # Warm up -> From 0 to 100% during first 10 epochs
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )

        # Main scheduler - will decrease LR as progressing thru epochs
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=(total_epochs - warmup_epochs)
        )

        # Use both schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # When run scheduler
                "frequency": 1,
            },
        }


def get_testing_transformation(input_size, img_mean, img_std):
    """Prepared transformation for validation and testing"""
    return T.Compose(
        [
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ]
    )
