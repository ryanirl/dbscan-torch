"""Border tiebreak: an exact-distance tie resolves to the core point with
the smallest ORIGINAL index, on both devices.

Geometry: two single-core clusters at (0, 0) and (2, 0), each propped up by
six coincident support points on its far side, and one non-core probe at
(1, 0) sitting at distance exactly 1.0 = eps from both cores (all
coordinates exact in float32). The probe's cluster must follow the input
order of the two cores -- not the neighbor-scan order, which always visits
the left cell first and would pin the probe to the left cluster regardless
of input order.
"""

import pytest
import torch

from dbscan_torch import dbscan2d

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

_EPS = 1.0
_MIN_SAMPLES = 8  # core a0: 6 support + itself + probe = 8; probe: 3 -> border

_SEGMENTS = {
    "a0": [(0.0, 0.0)],
    "a_support": [(-0.5, 0.0)] * 6,
    "b0": [(2.0, 0.0)],
    "b_support": [(2.5, 0.0)] * 6,
    "probe": [(1.0, 0.0)],
}


def _build(order: list[str]) -> tuple[torch.Tensor, dict[str, int]]:
    points: list[tuple[float, float]] = []
    first_index: dict[str, int] = {}
    for name in order:
        first_index[name] = len(points)
        points.extend(_SEGMENTS[name])
    return torch.tensor(points, dtype=torch.float32), first_index


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "order,winner",
    [
        (["a0", "a_support", "b0", "b_support", "probe"], "a0"),
        (["b0", "b_support", "a_support", "a0", "probe"], "b0"),
    ],
    ids=["a0_first", "b0_first"],
)
def test_tied_border_follows_smallest_input_index(device, order, winner):
    X, first_index = _build(order)
    labels = dbscan2d(X.to(device), eps=_EPS, min_samples=_MIN_SAMPLES).cpu()

    n_clusters = int(labels.max().item()) + 1
    assert n_clusters == 2, f"expected 2 clusters, got {n_clusters}"
    probe_label = int(labels[first_index["probe"]])
    assert probe_label != -1, "probe should be a border point, not noise"
    winner_label = int(labels[first_index[winner]])
    assert probe_label == winner_label, (
        f"tied border point should join {winner}'s cluster "
        f"(smallest original index), got the other side"
    )
