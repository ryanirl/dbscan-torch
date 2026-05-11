"""Use dbscan-torch as a drop-in for ``sklearn.cluster.DBSCAN``.

The ``DBSCAN`` class accepts numpy arrays and returns numpy arrays, matching
sklearn's API. Useful when you want to swap in dbscan-torch without touching
the surrounding pipeline.
"""

import numpy as np

from dbscan_torch import DBSCAN


def main() -> None:
    rng = np.random.default_rng(0)
    X = np.concatenate([
        rng.normal((0.0, 0.0), 0.3, size=(200, 2)),
        rng.normal((5.0, 0.0), 0.3, size=(200, 2)),
        rng.normal((2.5, 4.0), 0.3, size=(200, 2)),
    ]).astype(np.float32)

    # Identical to sklearn.cluster.DBSCAN(eps=..., min_samples=...).fit(X).
    clf = DBSCAN(eps=1.0, min_samples=5).fit(X)

    print(f"labels.shape: {clf.labels_.shape}")
    print(f"labels.dtype: {clf.labels_.dtype}")
    print(f"clusters:     {int(clf.labels_.max()) + 1}")
    print(f"noise:        {int((clf.labels_ == -1).sum())}")


if __name__ == "__main__":
    main()
