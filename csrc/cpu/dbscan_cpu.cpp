// CPU torch wrapper: validate + unpack an (N, 2) float32 tensor, call into
// the grid implementation, return int32 labels (-1 = noise).

#include <torch/extension.h>

#include <vector>

#include "dbscan_grid.hpp"

namespace {

// Split interleaved (x, y) into two contiguous arrays. The grid implementation
// takes xs[]/ys[] separately. Cheap relative to DBSCAN itself.
inline void split_xy(const float* data, int32_t n, std::vector<float>& xs,
                     std::vector<float>& ys) {
  xs.resize(n);
  ys.resize(n);
  for (int32_t i = 0; i < n; ++i) {
    xs[i] = data[2 * i];
    ys[i] = data[2 * i + 1];
  }
}

}  // namespace

torch::Tensor dbscan2d_cpu(torch::Tensor X, double eps, int64_t min_samples) {
  TORCH_CHECK(X.is_cpu(), "dbscan2d_cpu expects a CPU tensor");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
  TORCH_CHECK(X.dim() == 2 && X.size(1) == 2, "X must have shape (N, 2)");

  X = X.contiguous();
  const int32_t n = static_cast<int32_t>(X.size(0));
  auto labels = torch::full({n}, -1, X.options().dtype(torch::kInt32));
  if (n == 0) return labels;

  std::vector<float> xs, ys;
  split_xy(X.data_ptr<float>(), n, xs, ys);

  dbscan_grid::Result result;
  {
    pybind11::gil_scoped_release release;
    result = dbscan_grid::dbscan2d(xs.data(), ys.data(), n,
                                   static_cast<float>(eps),
                                   static_cast<int32_t>(min_samples));
  }

  std::copy(result.labels.begin(), result.labels.end(),
            labels.data_ptr<int32_t>());
  return labels;
}
