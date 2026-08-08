"""Eps-boundary tests for the bounding-box pruning.

The dense-data pruning (per-cell point bounding boxes gating the neighbor
scans) is only sound if "farther than eps from the box" is never wrongly
concluded for a pair the unpruned scan would accept. That question lives
entirely in float32 rounding at the eps boundary, so these tests pin the
invariant directly: two core groups merge if and only if the float32
distance rule (ex * ex + ey * ey <= eps * eps, all float32) says their
closest pair is within eps.

Constructed cases place coincident point groups at exactly-eps and
one-ulp-off distances across a ring-2 cell offset -- the geometry the prune
exists for, where the boxes degenerate to single points and the prune's
verdict must agree with the scan's bit for bit. A brute-force parity test
then covers the boundary statistically on dense random data.
"""

import numpy as np
import pytest
import torch

from dbscan_torch import dbscan2d

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

_EPS = 1.0
_MIN_SAMPLES = 5


def _cell_of(x: float, y: float, eps: float) -> tuple[int, int]:
    """Replicate the implementation's cell assignment (float32 throughout)."""
    inv_r = np.float32(np.sqrt(np.float32(2.0))) / np.float32(eps)
    kx = int(np.floor(np.float32(x) * inv_r))
    ky = int(np.floor(np.float32(y) * inv_r))
    return kx, ky


def _float32_within_eps(a: tuple[float, float], b: tuple[float, float],
                        eps: float) -> bool:
    """The implementation's acceptance rule, replicated in numpy float32."""
    ex = np.float32(b[0]) - np.float32(a[0])
    ey = np.float32(b[1]) - np.float32(a[1])
    eps_sq = np.float32(eps) * np.float32(eps)
    return bool(ex * ex + ey * ey <= eps_sq)


def _pair_groups(a: tuple[float, float], b: tuple[float, float],
                 device: str) -> torch.Tensor:
    """min_samples coincident points at each location: both groups are core
    (a full cell), the bounding boxes degenerate to the points themselves,
    and the groups land in one cluster iff the a-b pair is within eps."""
    pts = np.array([a] * _MIN_SAMPLES + [b] * _MIN_SAMPLES, dtype=np.float32)
    return torch.from_numpy(pts).to(device)


_JUST_OVER = float(np.nextafter(np.float32(1.5), np.float32(2.0)))
_JUST_UNDER = float(np.nextafter(np.float32(1.5), np.float32(0.0)))


def _boundary_float(eps: float, cell: int, side: str) -> float:
    """The extreme float32 coordinate assigned to ``cell``: the largest
    ("high") or smallest ("low") x with floor(x * inv_r) == cell, found by
    nudging from the real-arithmetic boundary. Working in actual floats
    matters here: the real boundary itself can round into either cell."""
    inv_r = np.float32(np.sqrt(np.float32(2.0))) / np.float32(eps)
    x = np.float32(cell + (1 if side == "high" else 0)) / inv_r
    step = np.float32(-np.inf) if side == "high" else np.float32(np.inf)
    while int(np.floor(x * inv_r)) != cell:
        x = np.nextafter(x, step)
    while int(np.floor(np.nextafter(x, -step) * inv_r)) == cell:
        x = np.nextafter(x, -step)
    return float(x)


_CELL0_HIGH = _boundary_float(_EPS, 0, "high")
_CELL2_LOW = _boundary_float(_EPS, 2, "low")

# (a, b, required ring-2 x-offset). With eps=1 the cell side is 1/sqrt(2), so
# x=0.5 falls in cell 0 and x=1.5 in cell 2: a pair at distance exactly 1.0
# spans a ring-2 offset, the productive stencil edge the prune must not cut.
# The corner case probes the (2, 2) diagonal, whose geometric gap rounds to
# just over eps; whichever way float32 decides, implementation and rule must
# agree.
_CASES = [
    pytest.param((0.5, 0.5), (1.5, 0.5), id="exact_eps_ring2_axis"),
    pytest.param((0.5, 0.5), (_JUST_OVER, 0.5), id="one_ulp_over"),
    pytest.param((0.5, 0.5), (_JUST_UNDER, 0.5), id="one_ulp_under"),
    pytest.param(
        (_CELL0_HIGH, _CELL0_HIGH),
        (_CELL2_LOW, _CELL2_LOW),
        id="ring2_diagonal_corner",
    ),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("a,b", _CASES)
def test_merge_follows_float32_rule(device, a, b):
    kx_a, _ = _cell_of(*a, _EPS)
    kx_b, _ = _cell_of(*b, _EPS)
    assert kx_b - kx_a == 2, "construction drifted: pair no longer spans ring-2"

    X = _pair_groups(a, b, device)
    labels = dbscan2d(X, eps=_EPS, min_samples=_MIN_SAMPLES).cpu()

    expected_clusters = 1 if _float32_within_eps(a, b, _EPS) else 2
    n_clusters = len(set(labels.tolist()) - {-1})
    assert (labels >= 0).all(), "full cells of min_samples points must be core"
    assert n_clusters == expected_clusters, (
        f"pair at {a} - {b}: expected {expected_clusters} cluster(s), "
        f"got {n_clusters}"
    )


# ---------------------------------------------------------------------------
# Brute-force parity on dense random data: many near-eps pairs, no pruning.
# ---------------------------------------------------------------------------

def _bruteforce_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Reference DBSCAN with the implementation's exact rules: inclusive
    float32 eps, neighbor counts include self, borders take the nearest core
    with ties broken by smaller original index."""
    n = X.shape[0]
    ex = X[:, 0][None, :] - X[:, 0][:, None]
    ey = X[:, 1][None, :] - X[:, 1][:, None]
    d2 = ex * ex + ey * ey
    eps_sq = np.float32(eps) * np.float32(eps)
    within = d2 <= eps_sq
    is_core = within.sum(axis=1) >= min_samples

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    core_indices = np.flatnonzero(is_core)
    for i in core_indices:
        for j in np.flatnonzero(within[i] & is_core):
            root_i, root_j = find(int(i)), find(int(j))
            if root_i != root_j:
                parent[max(root_i, root_j)] = min(root_i, root_j)

    labels = np.full(n, -1, dtype=np.int32)
    root_to_id: dict[int, int] = {}
    for i in core_indices:
        root = find(int(i))
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        labels[i] = root_to_id[root]

    for i in np.flatnonzero(~is_core):
        candidates = np.flatnonzero(within[i] & is_core)
        if candidates.size:
            order = np.lexsort((candidates, d2[i, candidates]))
            labels[i] = labels[candidates[order[0]]]
    return labels


def _canonical(labels: np.ndarray) -> np.ndarray:
    out = np.full_like(labels, -1)
    seen: dict[int, int] = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        if label not in seen:
            seen[label] = len(seen)
        out[i] = seen[label]
    return out


@pytest.mark.parametrize("device", _DEVICES)
def test_bruteforce_parity_dense(device):
    """Overlapping blobs at a scale where borders bleed between clusters and
    many pairs sit near eps: any unsound prune shows up as a partition diff."""
    rng = np.random.default_rng(7)
    centers = rng.uniform(-4.0, 4.0, size=(8, 2)).astype(np.float32)
    parts = [
        rng.normal(loc=c, scale=1.0, size=(150, 2)).astype(np.float32)
        for c in centers
    ]
    X = np.concatenate(parts)
    rng.shuffle(X)

    ours = dbscan2d(torch.from_numpy(X).to(device), eps=0.35, min_samples=5)
    reference = _bruteforce_dbscan(X, eps=0.35, min_samples=5)

    diff = int((_canonical(ours.cpu().numpy()) != _canonical(reference)).sum())
    assert diff == 0, f"{diff}/{X.shape[0]} points disagree with brute force"
