from typing import Tuple

import numpy as np
from sklearn.cluster import HDBSCAN


def hdbscan(data: np.ndarray, min_cluster_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    data: shape (num samples, num features)
    """
    print(f"HDBSCAN clustering with min cluster size {min_cluster_size}")
    model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit(data)
    pred = model.labels_
    dist = 1 - model.probabilities_

    cluster_labels = sorted(np.unique(pred))
    num_clusters = (
        len(cluster_labels) - 1 if -1 in cluster_labels else len(cluster_labels)
    )
    print(f"Created {num_clusters} clusters")

    return pred, dist
