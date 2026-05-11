# Changelog

All notable changes to `dbscan_torch` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-10

### Added
- Build-time torch version is stamped into the package (generated
  `_build_info.py`). `__init__.py` checks it against the runtime
  `torch.__version__` on import and raises a clear, actionable `ImportError`
  if the major.minor differs, instead of the cryptic `undefined symbol:
  _ZN3c10...` from libc10.
- README guidance on installing alongside a pinned torch via
  `pip install --no-build-isolation dbscan-torch`.

### Fixed
- CI workflow and `[build-system].requires` now pin torch to the same range
  as `[project].dependencies` (`torch>=2.0,<2.8`). Prevents a build-vs-runtime
  ABI mismatch when pip's isolated build env would otherwise pull a newer
  torch than the runtime env.

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
