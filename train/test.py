import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pytorch_metric_learning import losses, miners
from shared import GenericReIDModel, ReIDLightningModel
from tqdm import tqdm

# All needed variable setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoints/reid-epoch=01-train_loss=3.26.ckpt"
TEST_DIR = "../datasets/VeRi/image_test/"
QUERY_IMAGE = "../datasets/VeRi/image_query/0027_c015_00011450_0.jpg"

# Use normal transformation for testing
test_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Load trained checkpoint
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


model = load_trained_model(CHECKPOINT_PATH, "resnet50", 576)

# Go thru test dir and extract vectors
image_names = [f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")]
gallery_features = []
gallery_names = []

with torch.no_grad():
    for name in tqdm(image_names):
        img_path = os.path.join(TEST_DIR, name)
        img = Image.open(img_path).convert("RGB")
        tensor = test_transform(img).unsqueeze(0).to(DEVICE)

        feature = model.model(tensor)
        # Normalizing vectors for cosine distance
        feature = F.normalize(feature, p=2, dim=1)

        gallery_features.append(feature)
        gallery_names.append(name)

# Join all vectors to one matrix
gallery_features = torch.cat(gallery_features, dim=0)

print(f"Looking for car: {QUERY_IMAGE}")
query_img = Image.open(QUERY_IMAGE).convert("RGB")
query_tensor = test_transform(query_img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    query_feature = model.model(query_tensor)
    query_feature = F.normalize(query_feature, p=2, dim=1)

# Matrix multiplication gives cosine distance, thanks to normalisation earlier
similarities = torch.mm(query_feature, gallery_features.t()).squeeze(0)

# Look for 10 most similiar vectors
top_values, top_indices = torch.topk(similarities, k=10)

# Print closes vectors
for i in range(0, 10):
    idx = top_indices[i].item()
    img_name = gallery_names[idx]
    score = top_values[i].item()

    print(f"Top {i}: {img_name} | Score = {score:.4f}")
