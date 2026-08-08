"""Determinism tests: identical input must produce bit-identical labels.

DBSCAN is a deterministic function of (X, eps, min_samples). The CUDA path
historically violated this through a write-write race between path-halving
find and the union-find flatten kernel (see docs/cuda-determinism-fix.md),
showing up as ~0.3-0.5% of labels changing between identical runs on dense
inputs. These tests pin the fix on both devices, using the input shapes that
surfaced it: overlapping dense clusters and near-percolation uniform noise.

Raw label equality (not partition equality) is intentional: the pipeline is
deterministic end to end, so even the arbitrary cluster ids must not move.
"""

import numpy as np
import pytest
import torch

from dbscan_torch import dbscan2d

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

_N = 50_000
_RUNS = 3


def _dense_overlapping_blobs(n: int, seed: int) -> torch.Tensor:
    """Gaussian blobs whose borders bleed into each other, so the union-find
    must merge across many cell boundaries -- the stress case for the race."""
    rng = np.random.default_rng(seed)
    n_clusters = 20
    centers = rng.uniform(-25.0, 25.0, size=(n_clusters, 2)).astype(np.float32)
    per = np.full(n_clusters, n // n_clusters)
    per[: n - per.sum()] += 1
    parts = [
        rng.normal(loc=centers[c], scale=2.0, size=(per[c], 2)).astype(np.float32)
        for c in range(n_clusters)
    ]
    pts = np.concatenate(parts)
    rng.shuffle(pts)
    return torch.from_numpy(pts)


def _near_percolation_uniform(n: int, seed: int) -> torch.Tensor:
    """Uniform noise at a density where eps=0.5, min_samples=5 sits near the
    percolation threshold: many small clusters with ambiguous borders, the
    regime where the original nondeterminism was easiest to observe."""
    rng = np.random.default_rng(seed)
    extent = 200.0 * np.sqrt(n / 1_000_000)
    return torch.from_numpy(
        rng.uniform(-extent, extent, size=(n, 2)).astype(np.float32)
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "make_input",
    [_dense_overlapping_blobs, _near_percolation_uniform],
    ids=["dense_blobs", "uniform"],
)
def test_repeated_runs_bit_identical(device, make_input):
    X = make_input(_N, seed=0).to(device)
    reference = dbscan2d(X, eps=0.5, min_samples=5)
    for _ in range(_RUNS - 1):
        labels = dbscan2d(X, eps=0.5, min_samples=5)
        assert torch.equal(reference, labels), (
            f"labels diverged across identical runs on {device}: "
            f"{(reference != labels).sum().item()} points differ"
        )
