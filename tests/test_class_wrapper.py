"""Tests for the sklearn-compatible ``DBSCAN`` class wrapper.

The functional API ``dbscan2d`` is exercised in test_basic.py and
test_sklearn_parity.py. This file checks the input-type-preservation contract
of the class wrapper:

  * numpy in -> numpy out (matches sklearn).
  * torch in -> torch out, on the same device (matches the functional API).
  * float64 / non-float32 input is auto-cast.
  * ``fit`` returns self, ``labels_`` is populated, ``fit_predict`` is a one-shot.
"""

import numpy as np
import pytest
import torch

from dbscan_torch import DBSCAN


def _three_blobs_np(n_per=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    return np.concatenate(
        [rng.normal(loc=c, scale=0.3, size=(n_per, 2)) for c in centers]
    ).astype(np.float32)


def test_numpy_in_numpy_out():
    X = _three_blobs_np()
    clf = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert isinstance(clf.labels_, np.ndarray)
    assert clf.labels_.shape == (X.shape[0],)
    n_clusters = len(set(clf.labels_.tolist()) - {-1})
    assert n_clusters == 3


def test_torch_in_torch_out_cpu():
    X = torch.from_numpy(_three_blobs_np())
    clf = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert isinstance(clf.labels_, torch.Tensor)
    assert clf.labels_.dtype == torch.int32
    assert clf.labels_.device == X.device


def test_fit_returns_self():
    X = _three_blobs_np()
    clf = DBSCAN(eps=1.0, min_samples=5)
    out = clf.fit(X)
    assert out is clf


def test_fit_predict_matches_fit_then_labels():
    X = _three_blobs_np()
    a = DBSCAN(eps=1.0, min_samples=5).fit_predict(X)
    b = DBSCAN(eps=1.0, min_samples=5).fit(X).labels_
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    np.testing.assert_array_equal(a, b)


def test_float64_torch_input_is_cast():
    """Class wrapper should auto-cast float64 input rather than erroring."""
    X = torch.from_numpy(_three_blobs_np()).to(torch.float64)
    clf = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert isinstance(clf.labels_, torch.Tensor)
    assert clf.labels_.shape == (X.shape[0],)


def test_y_argument_ignored():
    """sklearn parity: ``fit(X, y)`` accepts and ignores ``y``."""
    X = _three_blobs_np()
    y = np.zeros(X.shape[0], dtype=np.int64)
    a = DBSCAN(eps=1.0, min_samples=5).fit_predict(X, y)
    b = DBSCAN(eps=1.0, min_samples=5).fit_predict(X)
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_cuda_in_cuda_out():
    X = torch.from_numpy(_three_blobs_np()).cuda()
    clf = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert isinstance(clf.labels_, torch.Tensor)
    assert clf.labels_.is_cuda
