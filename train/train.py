"""Main training file"""

import os
import random
import sys

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.samplers import MPerClassSampler
from shared import GenericReIDModel, ReIDLightningModel, get_testing_transformation
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Dataset, Subset

# For faster learning
torch.set_float32_matmul_precision("medium")


class VeRiDataset(Dataset):
    """Specific implementation for VeRi dataset"""

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        self.transform = transform

        # Use IDs from file names and map them to `id_to_class`
        raw_ids = sorted(
            list(set([int(name.split("_")[0]) for name in self.img_names]))
        )
        self.id_to_class = {raw_id: i for i, raw_id in enumerate(raw_ids)}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        parts = img_name.split("_")

        # Get Vehicle ID and map it to the correct class label
        raw_id = int(parts[0])
        label = self.id_to_class[raw_id]

        # Get Camera ID (Strip the 'c' and convert to integer)
        cam_id = int(parts[1].replace("c", ""))

        if self.transform:
            image = self.transform(image)

        return image, label, cam_id


class VeRiDatasetSubset(Dataset):
    """Helper class to apply transformations to a subset of the dataset"""

    def __init__(self, whole_dataset, subset_indices, transform, label_map=None):
        self.subset = Subset(whole_dataset, subset_indices)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label, cam_id = self.subset[idx]
        if self.transform:
            image = self.transform(image)

        if self.label_map is not None:
            label = self.label_map[label]

        return image, label, cam_id


def get_veri_split(train_dataset: VeRiDataset, veri_percent=0.1, seed=42):
    """Helper function to get a random subset of the VeRi dataset for validation"""
    all_indices = list(train_dataset.id_to_class.keys())
    random.seed(seed)
    random.shuffle(all_indices)

    split_point = int(len(all_indices) * (1.0 - veri_percent))
    train_ids = all_indices[:split_point]
    val_ids = all_indices[split_point:]

    train_indices = [
        i
        for i, img_name in enumerate(train_dataset.img_names)
        if int(img_name.split("_")[0]) in train_ids
    ]

    val_indices = [
        i
        for i, img_name in enumerate(train_dataset.img_names)
        if int(img_name.split("_")[0]) in val_ids
    ]

    # while id_to_class maps raw Vehicle IDs -> Global Class
    # but because some Global Classes went to validation set, we need to remap to continuous mapping
    # train_label_map maps Global Class from VeRiDataset -> Train Class (which accounts for the Classes that went to validation set)
    train_label_map = {
        train_dataset.id_to_class[raw_id]: i
        for i, raw_id in enumerate(sorted(train_ids))
    }

    return train_indices, val_indices, train_label_map


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <checkpoints_path> <model_name>")
        sys.exit(1)

    CHECKPOINTS_PATH = sys.argv[1]
    MODEL_NAME = sys.argv[2]

    # Number of ids in VeRi train dataset
    full_train_dataset = VeRiDataset(
        img_dir="../datasets/VeRi/image_train/", transform=None
    )
    VAL_PERCENT = 0.1

    # Get model supported resolution and compose transformations
    data_config = resolve_data_config({}, model=MODEL_NAME)

    input_size = data_config["input_size"][1:]
    img_mean = data_config["mean"]
    img_std = data_config["std"]

    # TODO: Do more research on transformations
    train_transform = T.Compose(
        [
            T.Resize(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ]
    )

    validation_transform = get_testing_transformation(
        input_size=input_size, img_mean=img_mean, img_std=img_std
    )

    train_subset_indices, val_subset_indices, train_label_map = get_veri_split(
        full_train_dataset, veri_percent=VAL_PERCENT
    )

    model = GenericReIDModel(MODEL_NAME)
    train_dataset = VeRiDatasetSubset(
        whole_dataset=full_train_dataset,
        subset_indices=train_subset_indices,
        transform=train_transform,
        label_map=train_label_map,
    )
    val_dataset = VeRiDatasetSubset(
        whole_dataset=full_train_dataset,
        subset_indices=val_subset_indices,
        transform=validation_transform,
    )

    NUM_VERI_TRAIN_CLASSES = len(train_label_map)
    print(f"Number of classes in training set: {NUM_VERI_TRAIN_CLASSES}")

    # Create criterion metrics with number of classes matching number of ids in VeRi train set
    # Using [NormalizedSoftmaxLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#normalizedsoftmaxloss)
    criterion_metric = losses.NormalizedSoftmaxLoss(
        num_classes=NUM_VERI_TRAIN_CLASSES, embedding_size=model.feature_dim
    )

    miner = miners.MultiSimilarityMiner()

    batch_size = 200
    loader_workers = 24

    print("Extracting labels for MPerClassSampler...")
    train_labels = []
    for idx in train_subset_indices:
        img_name = full_train_dataset.img_names[idx]
        raw_id = int(img_name.split("_")[0])
        global_label = full_train_dataset.id_to_class[raw_id]
        mapped_label = train_label_map[global_label]
        train_labels.append(mapped_label)

    # m=4 means for every car include 4 images (50 different cars per batch for 200 batch size)
    sampler = MPerClassSampler(
        labels=train_labels,
        m=4,
        batch_size=batch_size,
        length_before_new_iter=len(train_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=loader_workers,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers
    )

    # Instead of checkpointing based on less, checkpoint based on validation mAP
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS_PATH,
        filename=f"reid-{MODEL_NAME}" + "-{epoch:02d}-{val_mAP:.4f}",
        save_top_k=3,
        monitor="val_mAP",
        mode="max",
    )

    # Early stop - stop training if the model did not imporved for x time
    early_stop_callback = EarlyStopping(
        monitor="val_mAP",
        min_delta=0.003,  # Limit what is called improvement
        patience=5,  # If the model did not improve 5 times -> stop
        verbose=True,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="gpu",
        precision="bf16-mixed",  # Not working with resnet
        min_epochs=15,
    )

    # Time to train!
    lightning_model = ReIDLightningModel(model, criterion_metric, miner)
    trainer.fit(lightning_model, train_loader, val_loader)
