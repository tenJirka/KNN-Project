import os
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pytorch_metric_learning import losses, miners
from shared import GenericReIDModel, ReIDLightningModel
from tqdm import tqdm
import sys
from util import compute_reid_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = "../datasets/VeRi/image_test/"
QUERY_DIR = "../datasets/VeRi/image_query/"

test_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


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


def extract_features(directory_path, desc_name):
    image_names = [f for f in os.listdir(directory_path) if f.endswith(".jpg")]
    features, v_ids, c_ids = [], [], []

    with (
        torch.no_grad() as _,
        tqdm(image_names, desc=f"Extracting {desc_name}", leave=True) as pbar,
    ):
        for name in pbar:
            img_path = os.path.join(directory_path, name)
            img = Image.open(img_path).convert("RGB")
            tensor = test_transform(img).unsqueeze(0).to(DEVICE)

            feature = model.model(tensor)
            feature = F.normalize(feature, p=2, dim=1)

            features.append(feature.cpu())

            vid, cid = parse_filename(name)
            v_ids.append(vid)
            c_ids.append(cid)

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
    if len(sys.argv) != 4:
        print(
            "Usage: python test.py <model_checkpoint_path> <model_name> <num_of_classes>"
        )
        sys.exit(1)

    CHECKPOINT_PATH = sys.argv[1]
    MODEL_NAME = sys.argv[2]
    NUM_OF_CLASSES = int(sys.argv[3])
    model = load_trained_model(CHECKPOINT_PATH, MODEL_NAME, NUM_OF_CLASSES)

    gallery_features, gallery_vids, gallery_cids = extract_features(TEST_DIR, "Gallery")
    query_features, query_vids, query_cids = extract_features(QUERY_DIR, "Queries")

    mAP, cmc = evaluate_metrics(
        query_features,
        query_vids,
        query_cids,
        gallery_features,
        gallery_vids,
        gallery_cids,
    )

    print("\n--- Final VeRi Benchmark Results ---")
    print(f"mAP:    {mAP:.2%}")
    print(f"Rank-1: {cmc[0]:.2%}")
    print(f"Rank-5: {cmc[4]:.2%}")
    print(f"Rank-10:{cmc[9]:.2%}")
