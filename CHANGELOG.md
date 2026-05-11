# Changelog

All notable changes to `dbscan_torch` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-07

Initial release.

### Added
- `dbscan2d(X, eps, min_samples)` functional API. float32 `(N, 2)` tensor in,
  int32 `(N,)` labels out, on the same device as the input.
- `DBSCAN(eps, min_samples)` sklearn-compatible class wrapper. NumPy in to
  NumPy out (matches sklearn). Tensor in to Tensor out (matches the functional
  API).
- Parallel CPU implementation via `at::parallel_for` (thread count follows
  `torch.get_num_threads()`).
- CUDA implementation: keys, sort, cell-level union-find, labels all stay on
  device, no implicit host/device copies.
- sklearn-parity tests across blobs, moons, and circles datasets (ARI >= 0.99).
- CPU/CUDA bit-parity test on synthetic data.
- PEP 561 typing marker (`py.typed`) with inline annotations.

[Unreleased]: https://github.com/ryanirl/dbscan-torch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ryanirl/dbscan-torch/releases/tag/v0.1.0
