"""Main training file"""

import os
import random
import sys

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from dataset import PKUVehicleIdDataset, VeRiDataset, VeRiDatasetSubset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.samplers import MPerClassSampler
from shared import (
    GenericReIDModel,
    ReIDLightningModel,
    get_testing_transformation,
    parse_checkpoint_filename,
)
from timm.data import resolve_data_config
from torch.utils.data import ConcatDataset, DataLoader

# For faster learning
torch.set_float32_matmul_precision("medium")


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
    if len(sys.argv) not in [3, 4]:
        print("Usage: python train.py <checkpoints_path> <model_name> [--fit | --all]")
        sys.exit(1)

    use_fit = False
    use_all = False
    if len(sys.argv) == 4:
        if sys.argv[3] == "--fit":
            use_fit = True
        elif sys.argv[3] == "--all":
            use_all = True
        else:
            print(
                "Usage: python train.py <checkpoints_path> <model_name> [--fit | --all]"
            )
            sys.exit(1)

    CHECKPOINTS_PATH = sys.argv[1]
    MODEL_NAME = sys.argv[2]
    use_pretrained_checkpoint = False

    if "/" in MODEL_NAME:
        use_pretrained_checkpoint = True
        MODEL_PATH = MODEL_NAME
        MODEL_NAME, _, _, _ = parse_checkpoint_filename(os.path.basename(MODEL_NAME))

    if use_fit or use_all:
        full_base_dataset = VeRiDataset(
            img_dir="../datasets/fit/image_train/", transform=None
        )
    else:
        full_base_dataset = VeRiDataset(
            img_dir="../datasets/VeRi/image_train/", transform=None
        )
    VAL_PERCENT = 0.1

    model = GenericReIDModel(MODEL_NAME)

    # Get model supported resolution and compose transformations
    data_config = resolve_data_config({}, model=model.backbone)

    input_size = data_config["input_size"][1:]
    img_mean = data_config["mean"]
    img_std = data_config["std"]

    # TODO: Do more research on transformations
    train_transform = T.Compose(
        [
            T.Resize(input_size),
            T.RandomHorizontalFlip(p=0.5),
            # Random color change
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
            # Erase part of picture
            T.RandomErasing(
                p=0.5,  # Chance replacement will appear
                scale=(0.02, 0.4),  # 2%-40% of image
                ratio=(0.3, 3.3),
                value="random",  # Fill with random...
            ),
        ]
    )

    validation_transform = get_testing_transformation(
        input_size=input_size, img_mean=img_mean, img_std=img_std
    )

    train_subset_indices, val_subset_indices, train_label_map = get_veri_split(
        full_base_dataset, veri_percent=VAL_PERCENT
    )

    NUM_BASE_TRAIN_CLASSES = len(train_label_map)
    print(f"Number of classes in base training set: {NUM_BASE_TRAIN_CLASSES}")

    base_train_subset = VeRiDatasetSubset(
        whole_dataset=full_base_dataset,
        subset_indices=train_subset_indices,
        transform=train_transform,
        label_map=train_label_map,
    )

    train_datasets = [base_train_subset]
    NUM_CLASSES = NUM_BASE_TRAIN_CLASSES

    # Pokud použijeme --all, přidáme navíc celý VeRi dataset
    if use_all:
        full_veri_dataset = VeRiDataset(
            img_dir="../datasets/VeRi/image_train/", transform=None
        )
        # VeRi třídy posuneme tak, aby neolidovaly s FIT sadou
        veri_label_map = {
            global_class: global_class + NUM_CLASSES
            for raw_id, global_class in full_veri_dataset.id_to_class.items()
        }

        veri_train_subset = VeRiDatasetSubset(
            whole_dataset=full_veri_dataset,
            subset_indices=list(range(len(full_veri_dataset.img_names))),
            transform=train_transform,
            label_map=veri_label_map,
        )
        train_datasets.append(veri_train_subset)
        NUM_VERI_CLASSES = len(full_veri_dataset.id_to_class)
        print(f"Number of classes in additional VeRi dataset: {NUM_VERI_CLASSES}")
        NUM_CLASSES += NUM_VERI_CLASSES

    if not use_fit:
        # Původní logika bez flagů a nebo --all (PKU dataset se přidává)
        pku_train_dataset = PKUVehicleIdDataset(
            img_dir="../datasets/VehicleID_V1.0/image/",
            train_list_txt="../datasets/VehicleID_V1.0/train_test_split/train_list.txt",
            transform=train_transform,
            label_offset=NUM_CLASSES,
        )

        train_datasets.append(pku_train_dataset)
        NUM_PKU_TRAIN_CLASSES = len(pku_train_dataset.id_to_class)
        print(f"Number of classes in PKU dataset: {NUM_PKU_TRAIN_CLASSES}")
        NUM_CLASSES += NUM_PKU_TRAIN_CLASSES

    train_dataset = ConcatDataset(train_datasets)
    print(f"Total number of classes: {NUM_CLASSES}")

    val_dataset = VeRiDatasetSubset(
        whole_dataset=full_base_dataset,
        subset_indices=val_subset_indices,
        transform=validation_transform,
    )

    # Create criterion metrics with number of classes matching number of ids in VeRi train set
    # Using [NormalizedSoftmaxLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#normalizedsoftmaxloss)
    criterion_metric = losses.NormalizedSoftmaxLoss(
        num_classes=NUM_CLASSES, embedding_size=model.feature_dim
    )

    miner = miners.MultiSimilarityMiner()

    batch_size = 64
    loader_workers = 24

    print("Extracting labels for MPerClassSampler...")
    train_labels = []

    # Labels for the base train dataset
    for idx in train_subset_indices:
        img_name = full_base_dataset.img_names[idx]
        raw_id = int(img_name.split("_")[0])
        global_label = full_base_dataset.id_to_class[raw_id]
        mapped_label = train_label_map[global_label]
        train_labels.append(mapped_label)

    # Labels for VeRi dataset if --all
    if use_all:
        for img_name in full_veri_dataset.img_names:
            raw_id = int(img_name.split("_")[0])
            global_label = full_veri_dataset.id_to_class[raw_id]
            train_labels.append(veri_label_map[global_label])

    # Labels for PKU dataset if appended (oprava chybějících labelů v původním kódu)
    if not use_fit:
        print("Extracting PKU labels (this may take a moment)...")
        original_transform = pku_train_dataset.transform
        pku_train_dataset.transform = None
        for i in range(len(pku_train_dataset)):
            train_labels.append(pku_train_dataset[i][1])
        pku_train_dataset.transform = original_transform

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
        filename=f"reid-{MODEL_NAME}-c={NUM_CLASSES}" + "-{epoch:02d}-{val_mAP:.4f}",
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
    if use_pretrained_checkpoint:
        lightning_model = ReIDLightningModel(model, criterion_metric, miner)

        checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
        state_dict = checkpoint["state_dict"]

        # Remove old loss function data
        keys_to_remove = [k for k in state_dict.keys() if "criterion_metric" in k]
        for k in keys_to_remove:
            state_dict.pop(k)

        lightning_model.load_state_dict(state_dict, strict=False)
    else:
        lightning_model = ReIDLightningModel(model, criterion_metric, miner)
    trainer.fit(lightning_model, train_loader, val_loader)
