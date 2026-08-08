"""Wide-grid fallback tests.

Large inputs take a narrow-key fast path (16-bit offset coords) only when
the occupied grid spans fewer than 2^16 cells per axis; wider grids must
fall back to 64-bit keys. Every eval dataset fits the narrow path, so
without these tests the fallback branch of the large-N pipeline would only
ever be exercised by small inputs (which skip the range detection
entirely). The datasets here exceed the detection thresholds (512k on CUDA,
4M on CPU) while spanning far more than 2^16 cells per axis.
"""

import numpy as np
import pytest
import torch

from dbscan_torch import dbscan2d

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

_N_CLUSTERS = 300
_EPS = 0.5


def _wide_blobs(n: int, seed: int) -> torch.Tensor:
    """Well-separated tight blobs across a huge extent: the grid spans
    ~3x10^5 cells per axis, far beyond the 16-bit narrow-key range."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-6e4, 6e4, size=(_N_CLUSTERS, 2)).astype(np.float32)
    per = n // _N_CLUSTERS
    parts = [
        rng.normal(loc=c, scale=0.4, size=(per, 2)).astype(np.float32)
        for c in centers
    ]
    pts = np.concatenate(parts)
    rng.shuffle(pts)
    return torch.from_numpy(pts)


def _check(X: torch.Tensor) -> None:
    labels = dbscan2d(X, eps=_EPS, min_samples=5)
    rerun = dbscan2d(X, eps=_EPS, min_samples=5)
    assert torch.equal(labels, rerun), "wide-grid fallback is nondeterministic"

    n_clusters = int(labels.max().item()) + 1
    noise_frac = float((labels.cpu() == -1).float().mean())
    assert n_clusters == _N_CLUSTERS, (
        f"expected {_N_CLUSTERS} clusters, got {n_clusters}"
    )
    assert noise_frac < 0.01, f"unexpected noise fraction {noise_frac:.2%}"


@pytest.mark.parametrize("device", _DEVICES)
def test_wide_grid_above_cuda_threshold(device):
    # 600k exceeds the CUDA detection threshold, so the CUDA run exercises
    # detect-then-fall-back; the CPU run covers the same input on the plain
    # 64-bit path.
    X = _wide_blobs(600_000, seed=3).to(device)
    _check(X)


def test_wide_grid_above_cpu_threshold():
    # 4.2M exceeds the CPU detection threshold: range detection runs, the
    # span check fails, and the at::sort fallback must still produce the
    # right partition. CPU-only; the CUDA variant of this branch is covered
    # above at a size CI-friendly for the GPU-less runners.
    X = _wide_blobs(4_200_000, seed=4)
    _check(X)
