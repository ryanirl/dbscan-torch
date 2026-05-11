from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from typing import Any
from typing import Optional
from typing import Union

import numpy as np
import torch

from . import _C

# Public input type for the DBSCAN class wrapper. The functional API
# (``dbscan2d``) takes a torch.Tensor only. The class wrapper also accepts
# numpy arrays for sklearn parity.
ArrayLike2D = Union[np.ndarray, torch.Tensor]

__all__ = ["dbscan2d", "DBSCAN", "ArrayLike2D"]

try:
    __version__ = _pkg_version("dbscan-torch")
except PackageNotFoundError:
    # Running from a source tree without an installed dist (e.g. a clone with
    # no `pip install -e .` yet). Fall back so attribute access doesn't blow up.
    __version__ = "0.0.0+unknown"


def dbscan2d(X: torch.Tensor, eps: float, min_samples: int) -> torch.Tensor:
    """Run DBSCAN on a 2D point set.

    Dispatches to CPU or CUDA based on ``X.device``. The output tensor is on
    the same device as the input. No host<->device copies happen for CUDA
    inputs: keys, sort, union-find, labels all stay on the GPU.

    Args:
        X: float32 tensor of shape (N, 2).
        eps: neighborhood radius (> 0).
        min_samples: minimum neighbors (incl. self) to be a core point (>= 1).

    Returns:
        int32 tensor of shape (N,). -1 indicates noise, otherwise a cluster id
        in ``[0, n_clusters)``.
    """
    return _C.dbscan2d(X, float(eps), int(min_samples))


class DBSCAN:
    """sklearn-compatible class wrapper around :func:`dbscan2d`.

    Mirrors :class:`sklearn.cluster.DBSCAN` for the subset of arguments and
    attributes we support. Drop-in for code that does
    ``DBSCAN(eps=..., min_samples=...).fit_predict(X)``.

    Not supported (vs sklearn): ``metric``, ``algorithm``, ``leaf_size``,
    ``p``, ``sample_weight``, ``n_jobs``. Inputs must be 2D
    (shape ``(N, 2)``). Use ``torch.set_num_threads`` instead of ``n_jobs``.

    Input type is preserved end-to-end: ``numpy.ndarray`` in ->
    ``numpy.ndarray`` out (matches sklearn). ``torch.Tensor`` in ->
    ``torch.Tensor`` out on the same device (matches the functional API).

    Args:
        eps: neighborhood radius (> 0).
        min_samples: minimum neighbors (incl. self) to be a core point (>= 1).
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_: Optional[ArrayLike2D] = None

    def fit(self, X: ArrayLike2D, y: Any = None) -> "DBSCAN":
        """Fit the clusterer on ``X``. Stores labels in ``self.labels_``.

        ``y`` is accepted and ignored (sklearn API parity).
        """
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            Xt = X if X.dtype == torch.float32 else X.to(torch.float32)
        else:
            Xt = torch.as_tensor(X, dtype=torch.float32)

        labels = dbscan2d(Xt, self.eps, self.min_samples)
        self.labels_ = labels if is_torch else labels.cpu().numpy()
        return self

    def fit_predict(self, X: ArrayLike2D, y: Any = None) -> ArrayLike2D:
        """Fit and return labels in one call. Mirrors sklearn's signature."""
        return self.fit(X, y).labels_  # type: ignore[return-value]


