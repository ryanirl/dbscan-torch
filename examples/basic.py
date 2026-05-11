"""Smallest possible dbscan-torch example.

Generate three Gaussian blobs in 2D, cluster them with DBSCAN, print a
summary. Runs entirely in torch -- swap ``device="cpu"`` for ``device="cuda"``
to run the same code on the GPU.
"""

import torch

from dbscan_torch import dbscan2d


def main() -> None:
    torch.manual_seed(0)
    device = "cpu"

    # Three well-separated 2D clusters of 200 points each.
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], device=device)
    X = centers.repeat_interleave(200, dim=0) + 0.3 * torch.randn(600, 2, device=device)

    labels = dbscan2d(X, eps=1.0, min_samples=5)

    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    n_noise = int((labels == -1).sum())
    print(f"input:    N={X.shape[0]}, device={X.device}")
    print(f"output:   dtype={labels.dtype}, device={labels.device}")
    print(f"clusters: {n_clusters}")
    print(f"noise:    {n_noise}")


if __name__ == "__main__":
    main()
