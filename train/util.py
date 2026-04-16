import torchreid
import torch
import numpy as np


# Calculates mAP and CMC
def compute_reid_metrics(
    query_features,
    query_vids,
    query_cids,
    gallery_features,
    gallery_vids,
    gallery_cids,
    max_rank=50,
):
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    # Sanitize all IDs and Camera IDs to NumPy arrays
    query_vids = to_numpy(query_vids)
    query_cids = to_numpy(query_cids)
    gallery_vids = to_numpy(gallery_vids)
    gallery_cids = to_numpy(gallery_cids)

    # Sanitize features to ensure they are PyTorch tensors on CPU for similarity computation
    if not isinstance(query_features, torch.Tensor):
        query_features = torch.tensor(query_features)
    if not isinstance(gallery_features, torch.Tensor):
        gallery_features = torch.tensor(gallery_features)

    # Compute cosine similarity
    similarities = torch.mm(query_features, gallery_features.t()).cpu().numpy()

    # Convert cosine similarity to cosine distance
    distmat = 1.0 - similarities

    # Torchreid handles sorting, junk removal, CMC, and mAP calculation
    cmc, mAP = torchreid.metrics.rank.evaluate_rank(
        distmat=distmat,
        q_pids=query_vids,
        q_camids=query_cids,
        g_pids=gallery_vids,
        g_camids=gallery_cids,
        max_rank=max_rank,
    )

    return mAP, cmc
