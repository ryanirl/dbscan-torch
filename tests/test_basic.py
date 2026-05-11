"""Smoke + parity tests for dbscan_torch.

CPU tests run everywhere. CUDA-vs-CPU parity test is auto-skipped when CUDA is
unavailable, so the suite passes on CPU-only machines.
"""

import numpy as np
import pytest
import torch

from dbscan_torch import dbscan2d


def _three_blobs(n_per=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    pts = np.concatenate(
        [rng.normal(loc=c, scale=0.3, size=(n_per, 2)) for c in centers]
    ).astype(np.float32)
    return torch.from_numpy(pts)


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Map labels to a canonical form so cluster-id permutations compare equal.

    Noise (-1) stays -1. The first non-noise label encountered becomes 0, the
    next 1, and so on.
    """
    out = np.full_like(labels, -1)
    next_id = 0
    seen = {}
    for i, l in enumerate(labels):
        if l == -1:
            continue
        if l not in seen:
            seen[l] = next_id
            next_id += 1
        out[i] = seen[l]
    return out


def test_three_blobs_cpu():
    X = _three_blobs()
    labels = dbscan2d(X, eps=1.0, min_samples=5)
    assert labels.dtype == torch.int32
    assert labels.shape == (X.shape[0],)
    # Three well-separated blobs at scale 0.3 with eps=1 should yield three
    # clusters and very little noise.
    n_clusters = len(set(labels.tolist()) - {-1})
    assert n_clusters == 3, f"expected 3 clusters, got {n_clusters}"
    assert (labels == -1).sum().item() < 0.05 * X.shape[0]


def test_empty_cpu():
    X = torch.empty((0, 2), dtype=torch.float32)
    labels = dbscan2d(X, eps=1.0, min_samples=5)
    assert labels.shape == (0,)


def test_all_noise_cpu():
    # Spread points far enough that none have neighbors within eps.
    X = (torch.arange(20, dtype=torch.float32) * 100.0).unsqueeze(1).repeat(1, 2)
    labels = dbscan2d(X, eps=1.0, min_samples=2)
    assert (labels == -1).all()


def test_input_validation():
    with pytest.raises(RuntimeError, match=r"shape \(N, 2\)"):
        dbscan2d(torch.zeros((10, 3)), eps=1.0, min_samples=5)
    with pytest.raises(RuntimeError, match="float32"):
        dbscan2d(torch.zeros((10, 2), dtype=torch.float64), eps=1.0, min_samples=5)
    with pytest.raises(RuntimeError, match="eps"):
        dbscan2d(torch.zeros((10, 2)), eps=0.0, min_samples=5)
    with pytest.raises(RuntimeError, match="min_samples"):
        dbscan2d(torch.zeros((10, 2)), eps=1.0, min_samples=0)


def test_single_point_cpu():
    X = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    # min_samples=1 makes the lone point its own cluster (it's its own neighbor).
    labels = dbscan2d(X, eps=1.0, min_samples=1)
    assert labels.tolist() == [0]
    # min_samples=2 means it's noise.
    labels = dbscan2d(X, eps=1.0, min_samples=2)
    assert labels.tolist() == [-1]


def test_min_samples_exceeds_n_cpu():
    X = _three_blobs(n_per=10)
    labels = dbscan2d(X, eps=1.0, min_samples=10_000)
    assert (labels == -1).all()


def test_non_contiguous_input_cpu():
    """A sliced/transposed tensor still produces correct labels.

    The binding calls ``.contiguous()`` internally, so this should match the
    contiguous-input result exactly.
    """
    X = _three_blobs()
    # Take every other row -> non-contiguous view along dim 0.
    Xv = X[::2]
    Xc = Xv.contiguous().clone()
    assert not Xv.is_contiguous()
    a = dbscan2d(Xv, eps=1.0, min_samples=5)
    b = dbscan2d(Xc, eps=1.0, min_samples=5)
    assert torch.equal(a, b)


def test_output_device_matches_input_cpu():
    X = _three_blobs()
    labels = dbscan2d(X, eps=1.0, min_samples=5)
    assert labels.device == X.device


def test_sklearn_parity_cpu():
    sklearn = pytest.importorskip("sklearn.cluster")
    X = _three_blobs(n_per=300, seed=42)
    eps, ms = 0.6, 5
    ours = dbscan2d(X, eps=eps, min_samples=ms).cpu().numpy()
    theirs = sklearn.DBSCAN(eps=eps, min_samples=ms).fit_predict(X.numpy())
    # Compare canonical relabelings: cluster ids may differ, but partitions
    # should match up to border-tiebreak edge cases. Allow a tiny disagreement
    # margin for tied border points.
    diff = (_relabel(ours) != _relabel(theirs)).sum()
    assert diff <= 0.01 * X.shape[0], f"{diff} disagreements vs sklearn"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_gpu_parity():
    X = _three_blobs(n_per=500, seed=7)
    cpu_labels = dbscan2d(X, eps=0.6, min_samples=5)
    gpu_labels = dbscan2d(X.cuda(), eps=0.6, min_samples=5).cpu()
    diff = (_relabel(cpu_labels.numpy()) != _relabel(gpu_labels.numpy())).sum()
    assert diff <= 0.01 * X.shape[0], f"CPU/GPU disagree on {diff} points"
