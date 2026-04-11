"""Main training file"""

import os

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_metric_learning import losses, miners
from shared import GenericReIDModel, ReIDLightningModel
from torch.utils.data import DataLoader, Dataset

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


if __name__ == "__main__":
    # Number of ids in VeRi train dataset
    NUM_VERI_TRAIN_CLASSES = 576

    model = GenericReIDModel("resnet50")

    # Create criterion metrics with number of classes matching number of ids in VeRi train set
    # Using [NormalizedSoftmaxLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#normalizedsoftmaxloss)
    criterion_metric = losses.NormalizedSoftmaxLoss(
        num_classes=NUM_VERI_TRAIN_CLASSES, embedding_size=model.feature_dim
    )

    miner = miners.MultiSimilarityMiner()

    # TODO: Do more research on transformations
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load train split from VeRi dataset
    train_dataset = VeRiDataset(
        img_dir="../datasets/VeRi/image_train/", transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="reid-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        # precision="bf16-mixed",  # Not working as expected on my RX 9070 XT, disabling for now
    )

    # Time to train!
    lightning_model = ReIDLightningModel(model, criterion_metric, miner)
    trainer.fit(lightning_model, train_loader)
