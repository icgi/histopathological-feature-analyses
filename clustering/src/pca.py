from pathlib import Path
from typing import Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_dimensions(
    data: np.ndarray,
    targets: Optional[np.ndarray] = None,
    class_labels: Optional[Sequence[str]] = None,
    include_indices: Optional[Sequence[str]] = None,
    output_path: Optional[Path] = None,
):
    """
    data shape: (num samples, num features)
    targets: class label indices (num samples)
    class_labels: class label names (num classes)
    include_indices: Give list of class indices to include. Otherwise, all
    """

    if include_indices is not None:
        mask = np.isin(targets, include_indices)
        data = data[mask, :]
        targets = targets[mask]
        class_labels = [class_labels[i] for i in include_indices]

    print("Principal component analysis")
    print(f"Input samples:    {data.shape[0]:>6}")
    print(f"Input dimensions: {data.shape[1]:>6}")
    pca = PCA(n_components=2)
    features = pca.fit(data).transform(data)

    # Percentage of variance explained for each components
    print(
        "Explained variance ratio (first two components): "
        f"{pca.explained_variance_ratio_}"
    )

    plt.figure()
    if targets is None:
        plt.scatter(features[:, 0], features[:, 1], alpha=0.8, lw=2)
    else:
        class_indices = sorted(set(targets))
        class_labels = class_indices if class_labels is None else class_labels
        colors = list(mcolors.TABLEAU_COLORS.values())[: len(class_labels)]
        assert isinstance(targets, np.ndarray)
        print(f"{'Index':<6}{'Class':<30}{'Samples':>7}")
        for c, i, name in zip(colors, class_indices, class_labels):
            print(f"{i:<6}{name:<30}{sum(targets == i):>7}")
            plt.scatter(
                features[targets == i, 0],
                features[targets == i, 1],
                color=c,
                alpha=0.8,
                lw=2,
                label=name,
            )
        plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA")

    if output_path is not None:
        # output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight", transparent=False)
    else:
        plt.show()


if __name__ == "__main__":
    # Debug
    from sklearn import datasets

    iris = datasets.load_iris()
    plot_dimensions(iris.data, targets=iris.target, class_labels=iris.target_names)
