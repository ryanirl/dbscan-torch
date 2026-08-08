// CPU torch wrapper: validate + unpack an (N, 2) float32 tensor, call into
// the grid implementation, return int32 labels (-1 = noise).

#include <torch/extension.h>

#include <map>
#include <string>
#include <utility>
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

namespace {

// Shared core: runs the grid algorithm, returning labels. Optionally fills
// per-stage timings when ``timings`` is non-null. The split-xy + label copy
// are CPU torch boilerplate that doesn't belong in the timed region.
inline torch::Tensor run_cpu(torch::Tensor X, double eps, int64_t min_samples,
                             dbscan_grid::Timings* timings) {
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
                                   static_cast<int32_t>(min_samples), timings);
  }

  std::copy(result.labels.begin(), result.labels.end(),
            labels.data_ptr<int32_t>());
  return labels;
}

}  // namespace

torch::Tensor dbscan2d_cpu(torch::Tensor X, double eps, int64_t min_samples) {
  return run_cpu(std::move(X), eps, min_samples, /*timings=*/nullptr);
}

std::pair<torch::Tensor, std::map<std::string, float>>
dbscan2d_cpu_profile(torch::Tensor X, double eps, int64_t min_samples) {
  dbscan_grid::Timings timings;
  auto labels = run_cpu(std::move(X), eps, min_samples, &timings);
  std::map<std::string, float> out;
  for (int i = 0; i < dbscan_grid::Timings::N; ++i) {
    out[dbscan_grid::kStageNames[i]] = static_cast<float>(timings.ms[i]);
  }
  return {labels, out};
}
