"""Parity tests vs sklearn DBSCAN.

Each (dataset, eps, min_samples) case is parametrized over CPU and (when
available) CUDA. We assert:

  * exact cluster count match (border tiebreak shouldn't change the partition
    structure -- only which cluster individual border points belong to).
  * ARI >= 0.99 vs sklearn (border tiebreak can disagree on a small fraction
    of points, a tighter threshold would be flaky on dense data).

GPU-parametrized cases auto-skip on CPU-only machines.
"""

import numpy as np
import pytest
import torch

from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score

from dbscan_torch import dbscan2d


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _ds_blobs(n, k, std, seed):
    X, _ = make_blobs(n_samples=n, centers=k, cluster_std=std, random_state=seed)
    return X.astype(np.float32)


def _ds_moons(n, noise, seed):
    X, _ = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(np.float32)


def _ds_circles(n, noise, seed):
    X, _ = make_circles(n_samples=n, noise=noise, random_state=seed, factor=0.5)
    return X.astype(np.float32)


# (name, dataset_builder, eps, min_samples)
CASES = [
    ("blobs_500_dense",  lambda: _ds_blobs(500, 5, 0.5, 0),    0.50,  5),
    ("blobs_2k_sparse",  lambda: _ds_blobs(2000, 10, 1.5, 1),  0.80, 10),
    ("blobs_10k",        lambda: _ds_blobs(10000, 20, 0.8, 2), 0.50,  8),
    ("moons_1k",         lambda: _ds_moons(1000, 0.05, 0),     0.15,  5),
    ("moons_5k",         lambda: _ds_moons(5000, 0.07, 1),     0.10,  5),
    ("circles_2k",       lambda: _ds_circles(2000, 0.05, 0),   0.10,  5),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "name,builder,eps,min_samples",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_sklearn_parity(device, name, builder, eps, min_samples):
    X_np = builder()
    X = torch.from_numpy(X_np).to(device)

    ours = dbscan2d(X, eps=eps, min_samples=min_samples).cpu().numpy()
    theirs = SklearnDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_np)

    n_ours = int(ours.max()) + 1 if (ours >= 0).any() else 0
    n_theirs = int(theirs.max()) + 1 if (theirs >= 0).any() else 0
    assert n_ours == n_theirs, (
        f"[{name}/{device}] cluster count: ours={n_ours} sklearn={n_theirs}"
    )

    ari = adjusted_rand_score(theirs, ours)
    assert ari >= 0.99, f"[{name}/{device}] ARI {ari:.4f} below 0.99"


@pytest.mark.parametrize("device", _DEVICES)
def test_single_giant_cluster(device):
    """All points in one cluster -- exercises the cell-level UF heavily."""
    rng = np.random.default_rng(0)
    X_np = rng.normal(0.0, 0.3, size=(2000, 2)).astype(np.float32)
    X = torch.from_numpy(X_np).to(device)
    labels = dbscan2d(X, eps=0.5, min_samples=5).cpu().numpy()
    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    assert n_clusters == 1, f"expected 1 cluster, got {n_clusters}"
    assert (labels == -1).sum() == 0, "no points should be noise"


@pytest.mark.parametrize("device", _DEVICES)
def test_all_noise(device):
    """All points isolated -- exercises the no-core path."""
    X_np = (np.arange(50, dtype=np.float32) * 100.0)[:, None].repeat(2, axis=1)
    X = torch.from_numpy(X_np).to(device)
    labels = dbscan2d(X, eps=1.0, min_samples=2).cpu().numpy()
    assert (labels == -1).all(), "all points should be noise"
