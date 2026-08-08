# Changelog

All notable changes to `dbscan_torch` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-08

### Fixed
- CUDA `dbscan2d` is now deterministic. A write-write race in the cell-level
  union-find (path-halving `find` inside the flatten kernel racing other
  threads' halving writes) could leave a cell pointing at a non-root,
  changing cluster membership for ~0.3-0.5% of points between identical runs
  on dense inputs. The flatten kernel now uses a read-only find. Full
  investigation in `docs/cuda-determinism-fix.md`.

### Changed
- Dense-data performance: per-cell point bounding boxes now gate the
  neighbor-cell scans (core marking pass 2, cell union, border assignment),
  skipping scans that provably cannot contain a pair within eps. Runtime no
  longer depends materially on the data distribution: on an RTX 3090 at
  N=10M, overlapping blobs went from 22.1s to 12ms, an elongated single
  cluster from 2.5s to 10ms, dense blobs from 767ms to 10ms, and uniform
  from 41ms to 30ms. CPU improves 1.1-2.5x on the same dense cases.
- `k_unite_pairs` skips cell pairs already in the same component via a
  read-only root check before scanning.
- The CUDA neighbor-table kernel uses a flat thread-per-(cell, offset)
  mapping instead of one underfilled block per cell (previously the top
  stage on uniform data).

### Added
- `_C.dbscan2d_profile`: same dispatch as `dbscan2d`, additionally returns
  per-stage timings (CUDA events on GPU, wallclock on CPU) for
  attribution-driven optimization.
- `_C.dbscan2d_cuda_debug`: returns per-stage intermediate tensors for
  divergence debugging. Not part of the public API.
- Determinism regression tests (bit-identical labels across repeated runs on
  both devices) and eps-boundary tests pinning the pruning against the
  float32 distance rule, plus a brute-force parity test on dense data.

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

[Unreleased]: https://github.com/ryanirl/dbscan-torch/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ryanirl/dbscan-torch/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ryanirl/dbscan-torch/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ryanirl/dbscan-torch/releases/tag/v0.1.0
