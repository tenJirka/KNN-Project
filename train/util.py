import numpy as np
import torch


def compute_average_precision(sorted_matches):
    num_rel = sorted_matches.sum()
    if num_rel == 0:
        return 0.0

    tmp_cmc = sorted_matches.cumsum()
    precision_at_k = tmp_cmc / (np.arange(len(sorted_matches)) + 1)
    ap = (precision_at_k * sorted_matches).sum() / num_rel
    return ap


def compute_reid_metrics(
    query_features,
    query_vids,
    query_cids,
    gallery_features,
    gallery_vids,
    gallery_cids,
    max_rank=50,
):
    # Calculate cosine similarity using matrix multiplication
    similarities = torch.mm(query_features, gallery_features.t()).numpy()

    num_q, num_g = similarities.shape

    # Sort gallery items by highest similarity (descending order)
    indices = np.argsort(-similarities, axis=1)

    # Create a boolean matrix of matches (Query ID == Gallery ID)
    matches = (gallery_vids[indices] == query_vids[:, np.newaxis]).astype(np.int32)

    all_ap = []
    all_cmc = []

    for q_idx in range(num_q):
        q_vid = query_vids[q_idx]
        q_cid = query_cids[q_idx]
        order = indices[q_idx]

        # Remove gallery samples that have the same Vehicle ID and Camera ID as the query (junk images)
        remove_junk = (gallery_vids[order] == q_vid) & (gallery_cids[order] == q_cid)
        keep = ~remove_junk

        # Filter the matches list to drop the junk images
        raw_cmc = matches[q_idx][keep]

        # If there are no valid cross-camera gallery matches for this query, ignore it
        if not np.any(raw_cmc):
            continue

        # Compute CMC (Cumulative Matching Characteristics)
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = (
            1  # We only care if we found the correct car AT LEAST once by rank K
        )
        all_cmc.append(cmc[:max_rank])

        # Compute Average Precision (AP) for this query
        all_ap.append(compute_average_precision(raw_cmc))

    mAP = np.mean(all_ap)
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    mean_cmc = all_cmc.sum(0) / len(all_ap)

    return mAP, mean_cmc
