"""Plot DBSCAN output with matplotlib.

The most visual confirmation that the algorithm is doing what you expect:
generate a few blobs, cluster, color points by cluster id (with -1 = noise
shown in gray). Writes ``dbscan_example.png`` to the current directory.

Requires ``pip install matplotlib``.
"""

import matplotlib.pyplot as plt
import torch

from dbscan_torch import dbscan2d


def main() -> None:
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
    X = centers.repeat_interleave(300, dim=0) + 0.4 * torch.randn(900, 2)

    labels = dbscan2d(X, eps=1.0, min_samples=5).numpy()
    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    n_noise = int((labels == -1).sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot noise points in gray underneath, then cluster points colored on top.
    noise = labels == -1
    ax.scatter(X[noise, 0], X[noise, 1], c="lightgray", s=10, label="noise")
    ax.scatter(X[~noise, 0], X[~noise, 1], c=labels[~noise], cmap="tab10", s=10)

    ax.set_title(f"dbscan-torch: {n_clusters} clusters, {n_noise} noise")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
