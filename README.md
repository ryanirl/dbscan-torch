# dbscan-torch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GPU-accelerated 2D DBSCAN as a native PyTorch op. Tensor in, tensor out, on
the same device, with no host to device round-trip. Parallel CPU and CUDA
implementations, dispatched automatically on `X.device`.

Based on the grid-hashing approach from [Wang et al. (SIGMOD 2020)][wang-paper],
reimplemented as a native PyTorch extension with CUDA support.

## Install

    pip install dbscan-torch

CPU-only wheels work out of the box on Linux and macOS (not tested in Windows).
Building with CUDA requires the CUDA toolkit and a CUDA-enabled torch install at
build time.

### Installing alongside a pinned torch

The C++ extension is built against whatever torch lands in pip's isolated
build environment. If your runtime env has torch pinned to a version that
differs from "latest in the supported range," the build will compile
against the build-env torch and fail to import against your runtime torch
(PyTorch does not preserve binary compatibility across minor versions).
`dbscan-torch` detects this on import and tells you exactly what to do.
The fix is to build against your existing torch:

    pip install --no-build-isolation dbscan-torch

## Use

```python
import torch
from dbscan_torch import dbscan2d

X = torch.randn(100_000, 2, device="cuda")
labels = dbscan2d(X, eps=0.5, min_samples=10)
# labels: int32 tensor of shape (N,) on the same device as X.
# -1 = noise, otherwise a cluster id in [0, n_clusters).
```

X must be a float32 tensor of shape `(N, 2)`. The function dispatches to the
CPU or CUDA implementation automatically based on `X.device`. Thread count
for the CPU path follows `torch.get_num_threads()`, which is the same method that
controls parallelism for any other torch op.

See `examples/` for runnable scripts.

## Why this exists


`sklearn.cluster.DBSCAN` is single-threaded CPU and forces you to copy your
tensor off the device, cluster, then copy labels back. `dbscan-torch` keeps
the whole pipeline on the GPU while being magnitudes faster than sklearn and
lighter than [`dbscan-python`][dbscan-python-repo], while also supporting GPU
acceleration for massive gains on large N.

| Implementation | Parallelism | Output | Sweet Spot |
|---|---|---|---|
| `sklearn.cluster.DBSCAN` | single-threaded CPU | numpy | small N, prototyping |
| [cuml.DBSCAN][cuml-repo] (RAPIDS) | CUDA | cupy / numpy | RAPIDS ecosystem |
| [Wang et al.][wang-paper] (`pip install dbscan`) | parallel CPU | numpy | CPU at N < ~1M |
| `dbscan-torch` (this) | parallel CPU + CUDA | torch tensor | torch pipelines, CUDA at N > 70k, CPU at N > 1M |

The third row is the reference implementation of Wang et al.,
"Theoretically-Efficient and Practical Parallel DBSCAN" (SIGMOD 2020),
distributed as the [`dbscan-python`][dbscan-python-repo] package
(`pip install dbscan`).

[wang-paper]: https://dl.acm.org/doi/10.1145/3318464.3380582
[dbscan-python-repo]: https://github.com/wangyiqiu/dbscan-python
[cuml-repo]: https://github.com/rapidsai/cuml

## Scaling

<center>
    <img src="./imgs/dbscan_scaling.png" alt="scaling plot" width="100%">
</center>

Median time per call against N across all five implementations, on
synthetic 2D Gaussian clusters at constant density. CPU runs (sklearn,
Wang et al., `dbscan-torch` CPU) were measured locally on a MacBook Pro
Intel i9. GPU runs (cuml.DBSCAN, `dbscan-torch` CUDA) were measured on a
Modal T4 instance.

`sklearn` goes off-scale past 100k. `cuml.DBSCAN` is included for
reference but scales poorly on 2D inputs in this benchmark and caps out by
500k. The CUDA line stays under 100ms across the whole range while the CPU
implementations scale linearly. At 10M points `dbscan-torch` on CUDA is
~23x faster than Wang et al. and ~11x faster than the torch CPU path.


### Benchmark Details

| N          |   cuml | Wang et al. | torch CPU | torch CUDA | CUDA vs Wang et al. |
|------------|-------:|------------:|----------:|-----------:|--------------------:|
| 1,000      |  0.002 |      <0.001 |    <0.001 |      0.002 |      0.20x (slower) |
| 10,000     |  0.016 |       0.002 |     0.003 |      0.005 |      0.33x (slower) |
| 50,000     |  0.122 |       0.007 |     0.017 |      0.009 |      0.81x (slower) |
| 100,000    |  0.671 |       0.017 |     0.018 |      0.010 |              1.76x  |
| 500,000    | 18.700 |       0.068 |     0.071 |      0.006 |             11.21x  |
| 1,000,000  |      - |       0.150 |     0.124 |      0.017 |              8.85x  |
| 5,000,000  |      - |       0.947 |     0.456 |      0.064 |             14.80x  |
| 10,000,000 |      - |       2.159 |     1.013 |      0.093 |             23.17x  |

(- means run skipped. cuml caps out by 500k in this benchmark.)

## Author

Ryan Peters

## License

MIT. See [LICENSE](LICENSE).


