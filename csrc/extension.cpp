// Single Python-facing entry point.
//
// Validates the tensor up front, then routes to the CPU or CUDA implementation
// based on X.device. The CUDA arm is compiled in only when WITH_CUDA is defined
// (set by setup.py on the CUDA build path), so a CPU-only build has no
// unresolved symbol against dbscan2d_cuda.

#include <torch/extension.h>

torch::Tensor dbscan2d_cpu(torch::Tensor X, double eps, int64_t min_samples);

#ifdef WITH_CUDA
torch::Tensor dbscan2d_cuda(torch::Tensor X, double eps, int64_t min_samples);
#endif

static void check_input(const torch::Tensor& X, double eps, int64_t min_samples) {
  TORCH_CHECK(X.dim() == 2 && X.size(1) == 2, "X must have shape (N, 2)");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
  TORCH_CHECK(eps > 0.0, "eps must be > 0");
  TORCH_CHECK(min_samples >= 1, "min_samples must be >= 1");
}

torch::Tensor dbscan2d(torch::Tensor X, double eps, int64_t min_samples) {
  check_input(X, eps, min_samples);
  if (X.is_cuda()) {
#ifdef WITH_CUDA
    return dbscan2d_cuda(X.contiguous(), eps, min_samples);
#else
    TORCH_CHECK(false,
                "dbscan_torch was built without CUDA support. Rebuild with a "
                "CUDA-enabled torch and CUDA_HOME set, or pass a CPU tensor.");
#endif
  }
  return dbscan2d_cpu(X.contiguous(), eps, min_samples);
}

PYBIND11_MODULE(_C, m) {
  m.doc() = "Native PyTorch DBSCAN (2D, grid-hashed).";
  m.def("dbscan2d", &dbscan2d, pybind11::arg("X"), pybind11::arg("eps"),
        pybind11::arg("min_samples"),
        "Run DBSCAN on (N, 2) float32 tensor. Returns int32 labels "
        "(-1 = noise). Dispatches to CPU or CUDA based on X.device.");
}
