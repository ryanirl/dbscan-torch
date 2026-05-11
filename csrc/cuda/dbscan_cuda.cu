// CUDA implementation of grid-hashed 2D DBSCAN.
//
// Pipeline (all buffers live on device, we only copy the scalar num_cells to
// the host, since we need it to size num_cells-wide kernel launches):
//
//   1.  cell keys + perm                 (k_cell_keys)
//   2.  thrust::sort_by_key(keys, perm), gather sorted xs/ys (k_gather_xy)
//   3.  boundary flags + inclusive scan -> point_cell, num_cells, cell_start,
//       cell_keys                        (k_boundary_flags, scan, k_extract*)
//   4.  neighbor table via 25 binary searches per cell (k_build_ngh)
//   5.  core marking pass 1 (full cells), pass 2 (per-point sweep)
//                                        (k_core_pass1, k_core_pass2,
//                                         k_cell_status)
//   6.  cell-level union-find (atomic-CAS unite, path-halving find)
//                                        (k_uf_init, k_unite_pairs,
//                                         k_uf_flatten)
//   7.  cluster id renumbering via exclusive scan over root_flag
//                                        (k_root_flag, scan,
//                                         k_cluster_id_per_cell)
//   8.  core labels in sort order        (k_core_labels)
//   9.  border assignment with deterministic (d2, perm[j]) tiebreak
//                                        (k_border_assign)
//  10.  unpermute back to input order    (k_unpermute)
//
// See csrc/cpu/dbscan_grid.hpp for the reference CPU algorithm. The kernels
// mirror that file step for step. The one structural change is the
// hashmap-free neighbor lookup (binary search over cell_keys, which are
// already sorted by stage 2).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int32_t kNoise = -1;

__device__ __forceinline__ uint64_t pack_key(int32_t kx, int32_t ky) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(kx)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(ky));
}

// ============================================================================
// Stage 1: cell keys + perm
// ============================================================================
__global__ void k_cell_keys(const float* __restrict__ X, int32_t n,
                            float inv_r, uint64_t* __restrict__ keys,
                            int32_t* __restrict__ perm) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = X[2 * i + 0];
  float y = X[2 * i + 1];
  int32_t kx = static_cast<int32_t>(floorf(x * inv_r));
  int32_t ky = static_cast<int32_t>(floorf(y * inv_r));
  keys[i] = pack_key(kx, ky);
  perm[i] = i;
}

// ============================================================================
// Stage 2: gather sorted xs/ys via perm
// ============================================================================
__global__ void k_gather_xy(const float* __restrict__ X,
                            const int32_t* __restrict__ perm, int32_t n,
                            float* __restrict__ xs, float* __restrict__ ys) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int32_t p = perm[i];
  xs[i] = X[2 * p + 0];
  ys[i] = X[2 * p + 1];
}

// ============================================================================
// Stage 3a: boundary flags
// ============================================================================
__global__ void k_boundary_flags(const uint64_t* __restrict__ keys, int32_t n,
                                 int32_t* __restrict__ flag) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  flag[i] = (i == 0 || keys[i] != keys[i - 1]) ? 1 : 0;
}

// ============================================================================
// Stage 3b: point_cell from inclusive scan of flag
//   point_cell[i] = scan[i] - 1
// ============================================================================
__global__ void k_point_cell_from_scan(const int32_t* __restrict__ flag_scan,
                                       int32_t n,
                                       int32_t* __restrict__ point_cell) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  point_cell[i] = flag_scan[i] - 1;
}

// ============================================================================
// Stage 3c: extract cell_start (size num_cells+1) and cell_keys
// ============================================================================
__global__ void k_extract_cell_start(const int32_t* __restrict__ flag,
                                     const int32_t* __restrict__ flag_scan,
                                     const uint64_t* __restrict__ keys,
                                     int32_t n, int32_t num_cells,
                                     int32_t* __restrict__ cell_start,
                                     uint64_t* __restrict__ cell_keys) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (flag[i]) {
    int32_t c = flag_scan[i] - 1;
    cell_start[c] = i;
    cell_keys[c] = keys[i];
  }
  if (i == 0) cell_start[num_cells] = n;  // sentinel
}

// ============================================================================
// Stage 4: 5x5 neighbor table per cell, via binary search over sorted cell_keys.
// ============================================================================
__device__ int32_t binsearch_cell(const uint64_t* keys, int32_t num_cells,
                                  uint64_t target) {
  int32_t lo = 0, hi = num_cells;
  while (lo < hi) {
    int32_t mid = (lo + hi) >> 1;
    if (keys[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  if (lo < num_cells && keys[lo] == target) return lo;
  return -1;
}

__global__ void k_build_ngh(const uint64_t* __restrict__ cell_keys,
                            int32_t num_cells, int32_t* __restrict__ ngh) {
  int32_t c = blockIdx.x;
  int32_t off = threadIdx.x;
  if (c >= num_cells || off >= 25) return;
  uint64_t self_key = cell_keys[c];
  int32_t kx = static_cast<int32_t>(static_cast<uint32_t>(self_key >> 32));
  int32_t ky = static_cast<int32_t>(static_cast<uint32_t>(self_key));
  int32_t dx = (off / 5) - 2;
  int32_t dy = (off % 5) - 2;
  uint64_t target = pack_key(kx + dx, ky + dy);
  ngh[c * 25 + off] = binsearch_cell(cell_keys, num_cells, target);
}

// ============================================================================
// Stage 5a: pass 1 -- cells with size >= min_samples are fully core.
// ============================================================================
__global__ void k_core_pass1(const int32_t* __restrict__ cell_start,
                             int32_t num_cells, int32_t min_samples,
                             uint8_t* __restrict__ is_core) {
  int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_cells) return;
  int32_t s = cell_start[c], e = cell_start[c + 1];
  if (e - s >= min_samples) {
    for (int32_t i = s; i < e; ++i) is_core[i] = 1;
  }
}

// ============================================================================
// Stage 5b: pass 2 -- per-point neighbor count over the 5x5 cell window.
// ============================================================================
__global__ void k_core_pass2(const float* __restrict__ xs,
                             const float* __restrict__ ys,
                             const int32_t* __restrict__ cell_start,
                             const int32_t* __restrict__ point_cell,
                             const int32_t* __restrict__ ngh, int32_t n,
                             int32_t min_samples, float eps_sq,
                             uint8_t* __restrict__ is_core) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (is_core[i]) return;
  int32_t c = point_cell[i];
  float xi = xs[i], yi = ys[i];
  const int32_t* row = ngh + c * 25;
  int32_t cnt = 0;
  for (int32_t off = 0; off < 25; ++off) {
    int32_t nc = row[off];
    if (nc < 0) continue;
    int32_t s = cell_start[nc], e = cell_start[nc + 1];
    if (off == 12) {
      cnt += (e - s);
      if (cnt >= min_samples) break;
    } else {
      for (int32_t j = s; j < e; ++j) {
        float ex = xs[j] - xi, ey = ys[j] - yi;
        if (ex * ex + ey * ey <= eps_sq) ++cnt;
      }
      if (cnt >= min_samples) break;
    }
  }
  if (cnt >= min_samples) is_core[i] = 1;
}

// ============================================================================
// Stage 5c: per-cell core summary -- 0 (no core), 1 (mixed), 2 (fully core).
// ============================================================================
__global__ void k_cell_status(const uint8_t* __restrict__ is_core,
                              const int32_t* __restrict__ cell_start,
                              int32_t num_cells,
                              uint8_t* __restrict__ cell_status) {
  int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_cells) return;
  int32_t s = cell_start[c], e = cell_start[c + 1];
  bool any = false, all = (e > s);
  for (int32_t j = s; j < e; ++j) {
    if (is_core[j]) any = true;
    else all = false;
  }
  cell_status[c] = !any ? 0 : (all ? 2 : 1);
}

// ============================================================================
// Stage 6: union-find -- ECL-CC style.
//
// find(): path halving. The relaxed (non-atomic) write is safe because
// `gp = uf[p]` is always a valid ancestor of `p` (and thus of `x`). Concurrent
// writes by other threads can only redirect to higher ancestors, never to
// invalid nodes.
//
// unite(): atomicCAS on uf[max(a,b)] with expected == max(a,b) (i.e., "still
// its own root"). If the CAS fails another thread already redirected this
// root. Loop, re-find, retry. Roots only ever get redirected to lower ids.
// ============================================================================
__device__ __forceinline__ int32_t uf_find(int32_t* uf, int32_t x) {
  int32_t p = uf[x];
  while (p != x) {
    int32_t gp = uf[p];
    uf[x] = gp;  // path halving (relaxed)
    x = p;
    p = gp;
  }
  return x;
}

__device__ __forceinline__ void uf_unite(int32_t* uf, int32_t a, int32_t b) {
  while (true) {
    a = uf_find(uf, a);
    b = uf_find(uf, b);
    if (a == b) return;
    int32_t hi = a > b ? a : b;
    int32_t lo = a < b ? a : b;
    int32_t prev = atomicCAS(&uf[hi], hi, lo);
    if (prev == hi) return;
  }
}

__global__ void k_uf_init(int32_t* uf, int32_t num_cells) {
  int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < num_cells) uf[c] = c;
}

__global__ void k_unite_pairs(const float* __restrict__ xs,
                              const float* __restrict__ ys,
                              const int32_t* __restrict__ cell_start,
                              const uint8_t* __restrict__ is_core,
                              const uint8_t* __restrict__ cell_status,
                              const int32_t* __restrict__ ngh,
                              int32_t num_cells, float eps_sq,
                              int32_t* __restrict__ uf) {
  int32_t c = blockIdx.x;
  int32_t off = threadIdx.x;
  if (c >= num_cells || off >= 25 || off == 12) return;
  if (cell_status[c] == 0) return;
  int32_t nc = ngh[c * 25 + off];
  if (nc <= c) return;  // each undirected pair processed once (lower -> higher)
  if (cell_status[nc] == 0) return;

  int32_t s_c = cell_start[c], e_c = cell_start[c + 1];
  int32_t s_n = cell_start[nc], e_n = cell_start[nc + 1];
  bool found = false;
  for (int32_t i = s_c; i < e_c && !found; ++i) {
    if (!is_core[i]) continue;
    float xi = xs[i], yi = ys[i];
    for (int32_t j = s_n; j < e_n; ++j) {
      if (!is_core[j]) continue;
      float ex = xs[j] - xi, ey = ys[j] - yi;
      if (ex * ex + ey * ey <= eps_sq) {
        found = true;
        break;
      }
    }
  }
  if (found) uf_unite(uf, c, nc);
}

// Final compaction so uf[c] points directly to its root.
__global__ void k_uf_flatten(int32_t* uf, int32_t num_cells) {
  int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_cells) return;
  uf[c] = uf_find(uf, c);
}

// ============================================================================
// Stage 7: cluster id renumbering.
// ============================================================================
__global__ void k_root_flag(const int32_t* __restrict__ uf,
                            const uint8_t* __restrict__ cell_status,
                            int32_t num_cells,
                            int32_t* __restrict__ root_flag) {
  int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_cells) return;
  root_flag[c] = (uf[c] == c && cell_status[c] != 0) ? 1 : 0;
}

__global__ void k_cluster_id_per_cell(const int32_t* __restrict__ uf,
                                      const uint8_t* __restrict__ cell_status,
                                      const int32_t* __restrict__ root_id_scan,
                                      int32_t num_cells,
                                      int32_t* __restrict__ cluster_id) {
  int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_cells) return;
  if (cell_status[c] == 0) {
    cluster_id[c] = kNoise;
    return;
  }
  cluster_id[c] = root_id_scan[uf[c]];
}

// ============================================================================
// Stage 8: core labels in sort order.
// ============================================================================
__global__ void k_core_labels(const uint8_t* __restrict__ is_core,
                              const int32_t* __restrict__ point_cell,
                              const int32_t* __restrict__ cluster_id,
                              int32_t n,
                              int32_t* __restrict__ sorted_labels) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  sorted_labels[i] = is_core[i] ? cluster_id[point_cell[i]] : kNoise;
}

// ============================================================================
// Stage 9: border assignment.
//
// Tiebreak: lexicographic minimum of (d2, perm[j]) -- smaller d2 wins, ties go
// to the smaller original index. Each thread iterates sequentially over its
// candidate cores, so the result is a deterministic function of (xs, ys,
// perm, is_core).
// ============================================================================
__global__ void k_border_assign(
    const float* __restrict__ xs, const float* __restrict__ ys,
    const uint8_t* __restrict__ is_core,
    const int32_t* __restrict__ cell_start,
    const int32_t* __restrict__ point_cell,
    const int32_t* __restrict__ ngh,
    const uint8_t* __restrict__ cell_status,
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ sorted_labels_in, int32_t n, float eps_sq,
    int32_t* __restrict__ sorted_labels_out) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (is_core[i]) {
    sorted_labels_out[i] = sorted_labels_in[i];
    return;
  }
  int32_t c = point_cell[i];
  const int32_t* row = ngh + c * 25;
  float xi = xs[i], yi = ys[i];
  float best_d2 = eps_sq + 1.0f;
  int32_t best_perm = 0x7FFFFFFF;
  int32_t best_lab = kNoise;
  for (int32_t off = 0; off < 25; ++off) {
    int32_t nc = row[off];
    if (nc < 0 || cell_status[nc] == 0) continue;
    int32_t s = cell_start[nc], e = cell_start[nc + 1];
    for (int32_t j = s; j < e; ++j) {
      if (!is_core[j]) continue;
      float ex = xs[j] - xi, ey = ys[j] - yi;
      float d2 = ex * ex + ey * ey;
      if (d2 > eps_sq) continue;
      int32_t pj = perm[j];
      bool better = (d2 < best_d2) || (d2 == best_d2 && pj < best_perm);
      if (better) {
        best_d2 = d2;
        best_perm = pj;
        best_lab = sorted_labels_in[j];
      }
    }
  }
  sorted_labels_out[i] = best_lab;
}

// ============================================================================
// Stage 10: unpermute.
// ============================================================================
__global__ void k_unpermute(const int32_t* __restrict__ perm,
                            const int32_t* __restrict__ sorted_labels,
                            int32_t n, int32_t* __restrict__ labels) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  labels[perm[i]] = sorted_labels[i];
}

}  // namespace

torch::Tensor dbscan2d_cuda(torch::Tensor X, double eps, int64_t min_samples) {
  TORCH_CHECK(X.is_cuda(), "X must be on CUDA");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
  TORCH_CHECK(X.dim() == 2 && X.size(1) == 2, "X must be (N, 2)");
  X = X.contiguous();

  const int32_t n = static_cast<int32_t>(X.size(0));
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(X.device());
  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(X.device());
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(X.device());

  if (n == 0) return torch::empty({0}, opts_i32);

  const float fl_eps = static_cast<float>(eps);
  const float inv_r = std::sqrt(2.0f) / fl_eps;
  const float eps_sq = fl_eps * fl_eps;
  const int32_t ms = static_cast<int32_t>(min_samples);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto policy = thrust::cuda::par.on(stream);

  constexpr int kBlk = 256;
  auto grid = [](int32_t total) { return (total + kBlk - 1) / kBlk; };

  // ---------------------------------------------------------------- stage 1
  auto keys = torch::empty({n}, opts_i64);
  auto perm = torch::empty({n}, opts_i32);
  k_cell_keys<<<grid(n), kBlk, 0, stream>>>(
      X.data_ptr<float>(), n, inv_r,
      reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
      perm.data_ptr<int32_t>());

  // ---------------------------------------------------------------- stage 2
  thrust::sort_by_key(
      policy,
      thrust::device_pointer_cast(reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>())),
      thrust::device_pointer_cast(reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>())) + n,
      thrust::device_pointer_cast(perm.data_ptr<int32_t>()));

  auto xs = torch::empty({n}, X.options());
  auto ys = torch::empty({n}, X.options());
  k_gather_xy<<<grid(n), kBlk, 0, stream>>>(X.data_ptr<float>(),
                                            perm.data_ptr<int32_t>(), n,
                                            xs.data_ptr<float>(),
                                            ys.data_ptr<float>());

  // ---------------------------------------------------------------- stage 3
  auto flag = torch::empty({n}, opts_i32);
  auto flag_scan = torch::empty({n}, opts_i32);

  k_boundary_flags<<<grid(n), kBlk, 0, stream>>>(
      reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()), n,
      flag.data_ptr<int32_t>());

  thrust::inclusive_scan(
      policy, thrust::device_pointer_cast(flag.data_ptr<int32_t>()),
      thrust::device_pointer_cast(flag.data_ptr<int32_t>()) + n,
      thrust::device_pointer_cast(flag_scan.data_ptr<int32_t>()));

  // num_cells = flag_scan[n-1] -- the scalar must be on the host to size grids.
  int32_t num_cells = 0;
  C10_CUDA_CHECK(cudaMemcpyAsync(&num_cells,
                                 flag_scan.data_ptr<int32_t>() + (n - 1),
                                 sizeof(int32_t), cudaMemcpyDeviceToHost,
                                 stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  auto point_cell = torch::empty({n}, opts_i32);
  k_point_cell_from_scan<<<grid(n), kBlk, 0, stream>>>(
      flag_scan.data_ptr<int32_t>(), n, point_cell.data_ptr<int32_t>());

  auto cell_start = torch::empty({num_cells + 1}, opts_i32);
  auto cell_keys = torch::empty({num_cells}, opts_i64);
  k_extract_cell_start<<<grid(n), kBlk, 0, stream>>>(
      flag.data_ptr<int32_t>(), flag_scan.data_ptr<int32_t>(),
      reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()), n, num_cells,
      cell_start.data_ptr<int32_t>(),
      reinterpret_cast<uint64_t*>(cell_keys.data_ptr<int64_t>()));

  // ---------------------------------------------------------------- stage 4
  auto ngh = torch::empty({num_cells, 25}, opts_i32);
  // 25 threads/block, num_cells blocks. Low occupancy per block, but each
  // binary search is short and the table is built once.
  k_build_ngh<<<num_cells, 32, 0, stream>>>(
      reinterpret_cast<uint64_t*>(cell_keys.data_ptr<int64_t>()), num_cells,
      ngh.data_ptr<int32_t>());

  // ---------------------------------------------------------------- stage 5
  auto is_core = torch::zeros({n}, opts_u8);
  k_core_pass1<<<grid(num_cells), kBlk, 0, stream>>>(
      cell_start.data_ptr<int32_t>(), num_cells, ms,
      is_core.data_ptr<uint8_t>());

  k_core_pass2<<<grid(n), kBlk, 0, stream>>>(
      xs.data_ptr<float>(), ys.data_ptr<float>(),
      cell_start.data_ptr<int32_t>(), point_cell.data_ptr<int32_t>(),
      ngh.data_ptr<int32_t>(), n, ms, eps_sq, is_core.data_ptr<uint8_t>());

  auto cell_status = torch::empty({num_cells}, opts_u8);
  k_cell_status<<<grid(num_cells), kBlk, 0, stream>>>(
      is_core.data_ptr<uint8_t>(), cell_start.data_ptr<int32_t>(), num_cells,
      cell_status.data_ptr<uint8_t>());

  // ---------------------------------------------------------------- stage 6
  auto uf = torch::empty({num_cells}, opts_i32);
  k_uf_init<<<grid(num_cells), kBlk, 0, stream>>>(uf.data_ptr<int32_t>(),
                                                  num_cells);

  k_unite_pairs<<<num_cells, 32, 0, stream>>>(
      xs.data_ptr<float>(), ys.data_ptr<float>(),
      cell_start.data_ptr<int32_t>(), is_core.data_ptr<uint8_t>(),
      cell_status.data_ptr<uint8_t>(), ngh.data_ptr<int32_t>(), num_cells,
      eps_sq, uf.data_ptr<int32_t>());

  k_uf_flatten<<<grid(num_cells), kBlk, 0, stream>>>(uf.data_ptr<int32_t>(),
                                                     num_cells);

  // ---------------------------------------------------------------- stage 7
  auto root_flag = torch::empty({num_cells}, opts_i32);
  auto root_id_scan = torch::empty({num_cells}, opts_i32);
  k_root_flag<<<grid(num_cells), kBlk, 0, stream>>>(
      uf.data_ptr<int32_t>(), cell_status.data_ptr<uint8_t>(), num_cells,
      root_flag.data_ptr<int32_t>());

  thrust::exclusive_scan(
      policy, thrust::device_pointer_cast(root_flag.data_ptr<int32_t>()),
      thrust::device_pointer_cast(root_flag.data_ptr<int32_t>()) + num_cells,
      thrust::device_pointer_cast(root_id_scan.data_ptr<int32_t>()));

  auto cluster_id = torch::empty({num_cells}, opts_i32);
  k_cluster_id_per_cell<<<grid(num_cells), kBlk, 0, stream>>>(
      uf.data_ptr<int32_t>(), cell_status.data_ptr<uint8_t>(),
      root_id_scan.data_ptr<int32_t>(), num_cells,
      cluster_id.data_ptr<int32_t>());

  // ---------------------------------------------------------------- stage 8
  auto sorted_labels = torch::empty({n}, opts_i32);
  k_core_labels<<<grid(n), kBlk, 0, stream>>>(
      is_core.data_ptr<uint8_t>(), point_cell.data_ptr<int32_t>(),
      cluster_id.data_ptr<int32_t>(), n, sorted_labels.data_ptr<int32_t>());

  // ---------------------------------------------------------------- stage 9
  auto sorted_labels_final = torch::empty({n}, opts_i32);
  k_border_assign<<<grid(n), kBlk, 0, stream>>>(
      xs.data_ptr<float>(), ys.data_ptr<float>(), is_core.data_ptr<uint8_t>(),
      cell_start.data_ptr<int32_t>(), point_cell.data_ptr<int32_t>(),
      ngh.data_ptr<int32_t>(), cell_status.data_ptr<uint8_t>(),
      perm.data_ptr<int32_t>(), sorted_labels.data_ptr<int32_t>(), n, eps_sq,
      sorted_labels_final.data_ptr<int32_t>());

  // ---------------------------------------------------------------- stage 10
  auto labels = torch::full({n}, kNoise, opts_i32);
  k_unpermute<<<grid(n), kBlk, 0, stream>>>(
      perm.data_ptr<int32_t>(), sorted_labels_final.data_ptr<int32_t>(), n,
      labels.data_ptr<int32_t>());

  C10_CUDA_CHECK(cudaGetLastError());
  return labels;
}
