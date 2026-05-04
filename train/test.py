import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from dataset import ReIDTestDataset
from PIL import Image
from pytorch_metric_learning import losses, miners
from shared import (
    GenericReIDModel,
    ReIDLightningModel,
    determine_device,
    get_testing_transformation,
    parse_checkpoint_filename,
)
from timm.data import resolve_data_config
from tqdm import tqdm
from util import compute_reid_metrics

DEVICE = determine_device()

VERI_TEST_DIR = "../datasets/VeRi/image_test/"
VERI_QUERY_DIR = "../datasets/VeRi/image_query/"

FIT_TEST_DIR = "../datasets/fit/image_test/"
FIT_QUERY_DIR = "../datasets/fit/image_query/"


def load_trained_model(path, model_name, num_of_classes):
    model = GenericReIDModel(model_name)
    criterion_metric = losses.NormalizedSoftmaxLoss(num_of_classes, model.feature_dim)
    miner = miners.MultiSimilarityMiner()

    lightning_model = ReIDLightningModel.load_from_checkpoint(
        path, model=model, criterion_metric=criterion_metric, miner=miner
    )
    lightning_model.to(DEVICE)
    lightning_model.eval()
    return lightning_model


def parse_filename(filename):
    # VeRi filename format: 0027_c015_00011450_0.jpg
    # parts[0] = Vehicle ID (0027), parts[1] = Camera ID (c015)
    parts = filename.split("_")
    v_id = int(parts[0])
    c_id = int(parts[1].replace("c", ""))
    return v_id, c_id


def extract_features(
    directory_path, desc_name, test_transform, batch_size=64, workers=4
):
    dataset = ReIDTestDataset(directory_path, parse_filename, test_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    features, v_ids, c_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {desc_name}", leave=True):
            images, vids_batch, cids_batch = batch
            images = images.to(DEVICE, non_blocking=True)

            batch_features = model.model(images)
            batch_features = F.normalize(batch_features, p=2, dim=1)

            features.append(batch_features.cpu())
            v_ids.extend(vids_batch.numpy())
            c_ids.extend(cids_batch.numpy())

    return torch.cat(features, dim=0), np.array(v_ids), np.array(c_ids)


def evaluate_metrics(
    query_features,
    query_vids,
    query_cids,
    gallery_features,
    gallery_vids,
    gallery_cids,
    max_rank=50,
):
    return compute_reid_metrics(
        query_features,
        query_vids,
        query_cids,
        gallery_features,
        gallery_vids,
        gallery_cids,
        max_rank,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <model_checkpoint_path>")
        sys.exit(1)

    CHECKPOINT_PATH = sys.argv[1]
    MODEL_NAME, NUM_OF_CLASSES, _, _ = parse_checkpoint_filename(
        os.path.basename(CHECKPOINT_PATH)
    )

    model = GenericReIDModel(MODEL_NAME)

    # Get model supported resolution and compose transformations
    data_config = resolve_data_config({}, model=model.backbone)

    input_size = data_config["input_size"][1:]
    img_mean = data_config["mean"]
    img_std = data_config["std"]

    test_transform = get_testing_transformation(
        input_size=input_size, img_mean=img_mean, img_std=img_std
    )
    model = load_trained_model(CHECKPOINT_PATH, MODEL_NAME, NUM_OF_CLASSES)

    veri_gallery_features, veri_gallery_vids, veri_gallery_cids = extract_features(
        VERI_TEST_DIR, "Gallery", test_transform
    )
    veri_query_features, veri_query_vids, veri_query_cids = extract_features(
        VERI_QUERY_DIR, "Queries", test_transform
    )

    mAP, cmc = evaluate_metrics(
        veri_query_features,
        veri_query_vids,
        veri_query_cids,
        veri_gallery_features,
        veri_gallery_vids,
        veri_gallery_cids,
    )

    print("\n--- Final VeRi Benchmark Results ---")
    print(f"mAP:    {mAP:.2%}")
    print(f"Rank-1: {cmc[0]:.2%}")
    print(f"Rank-5: {cmc[4]:.2%}")
    print(f"Rank-10:{cmc[9]:.2%}")

    fit_gallery_features, fit_gallery_vids, fit_gallery_cids = extract_features(
        FIT_TEST_DIR, "Gallery", test_transform
    )
    fit_query_features, fit_query_vids, fit_query_cids = extract_features(
        FIT_QUERY_DIR, "Queries", test_transform
    )

    mAP, cmc = evaluate_metrics(
        fit_query_features,
        fit_query_vids,
        fit_query_cids,
        fit_gallery_features,
        fit_gallery_vids,
        fit_gallery_cids,
    )

    print("\n--- Final FIT Benchmark Results ---")
    print(f"mAP:    {mAP:.2%}")
    print(f"Rank-1: {cmc[0]:.2%}")
    print(f"Rank-5: {cmc[4]:.2%}")
    print(f"Rank-10:{cmc[9]:.2%}")
