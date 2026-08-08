// CUDA implementation of grid-hashed 2D DBSCAN.
//
// Pipeline (all buffers live on device, we only copy the scalar num_cells to
// the host, since we need it to size num_cells-wide kernel launches):
//
//   0.  (large inputs) grid extent for the narrow-key fast path
//                                        (k_grid_minmax)
//   1.  cell keys + perm                 (k_cell_keys; 32-bit offset packing
//                                         when the grid fits 2^16 per axis,
//                                         64-bit otherwise)
//   2.  cub radix sort with an explicit bit range, gather sorted xs/ys
//                                        (cub::DeviceRadixSort, k_gather_xy)
//   3.  boundary flags + inclusive scan -> point_cell, num_cells, cell_start,
//       cell_keys, per-cell bounding boxes
//                                        (k_boundary_flags, scan, k_extract*,
//                                         k_cell_bbox)
//   4.  neighbor table via 25 binary searches per cell (k_build_ngh)
//   5.  core marking pass 1 (full cells), pass 2 (per-point sweep,
//       center-first order, bbox-pruned) (k_core_pass1, k_core_pass2,
//                                         k_cell_status)
//   6.  cell-level union-find, two-phase (ring 1, flatten, ring 2), with
//       bbox pruning and already-united skip
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
// See csrc/cpu/dbscan_grid.hpp for the reference CPU algorithm. The stages
// correspond one to one, though the per-stage tactics differ where the
// hardware demands it (binary-search neighbor lookup here vs the CPU's
// merge-join; cub radix here vs the CPU's hand-rolled 2-pass radix).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cub/device/device_radix_sort.cuh>

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
// Stage 0 (large inputs only): kx/ky range, for the narrow-key fast path.
//
// When the occupied grid spans fewer than 2^16 cells per axis (any realistic
// dataset), the keys are packed as 16+16-bit offsets from (min_kx, min_ky)
// into the LOW 32 bits, and the radix sort runs over 32 instead of 64 bits,
// which halves its passes. Wide grids and small inputs (where the extra
// reduction + host sync would not pay) keep the original 64-bit packing.
// ============================================================================
__global__ void k_grid_minmax(const float* __restrict__ X, int32_t n,
                              float inv_r, int32_t* __restrict__ minmax) {
  __shared__ int32_t red[4];  // min_kx, max_kx, min_ky, max_ky
  if (threadIdx.x == 0) {
    red[0] = INT32_MAX;
    red[1] = INT32_MIN;
    red[2] = INT32_MAX;
    red[3] = INT32_MIN;
  }
  __syncthreads();
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int32_t kx = static_cast<int32_t>(floorf(X[2 * i + 0] * inv_r));
    int32_t ky = static_cast<int32_t>(floorf(X[2 * i + 1] * inv_r));
    atomicMin(&red[0], kx);
    atomicMax(&red[1], kx);
    atomicMin(&red[2], ky);
    atomicMax(&red[3], ky);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin(&minmax[0], red[0]);
    atomicMax(&minmax[1], red[1]);
    atomicMin(&minmax[2], red[2]);
    atomicMax(&minmax[3], red[3]);
  }
}

// ============================================================================
// Stage 1: cell keys + perm. ``shift`` selects the packing: 32 is the
// original (kx | ky as uint32 halves, mins are 0), 16 is the narrow path
// (16-bit offsets from the grid minimum, key fits in the low 32 bits).
// ============================================================================
__global__ void k_cell_keys(const float* __restrict__ X, int32_t n,
                            float inv_r, int32_t min_kx, int32_t min_ky,
                            int32_t shift, uint64_t* __restrict__ keys,
                            int32_t* __restrict__ perm) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = X[2 * i + 0];
  float y = X[2 * i + 1];
  int32_t kx = static_cast<int32_t>(floorf(x * inv_r));
  int32_t ky = static_cast<int32_t>(floorf(y * inv_r));
  uint32_t okx = static_cast<uint32_t>(kx - min_kx);
  uint32_t oky = static_cast<uint32_t>(ky - min_ky);
  keys[i] = (static_cast<uint64_t>(okx) << shift) | static_cast<uint64_t>(oky);
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
                            int32_t num_cells, int32_t shift,
                            int32_t* __restrict__ ngh) {
  int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(num_cells) * 25) return;
  int32_t c = static_cast<int32_t>(t / 25);
  int32_t off = static_cast<int32_t>(t % 25);
  uint64_t self_key = cell_keys[c];
  int64_t okx = static_cast<int64_t>(self_key >> shift);
  int64_t oky = static_cast<int64_t>(self_key & ((1ull << shift) - 1));
  int64_t tx = okx + (off / 5) - 2;
  int64_t ty = oky + (off % 5) - 2;

  uint64_t target;
  if (shift == 16) {
    // Narrow packing: offsets are 16-bit, so a neighbor coordinate outside
    // [0, 2^16) cannot exist (and letting it wrap would corrupt the okx
    // field of the packed target).
    if (tx < 0 || tx >= 65536 || ty < 0 || ty >= 65536) {
      ngh[c * 25 + off] = -1;
      return;
    }
    target = (static_cast<uint64_t>(tx) << 16) | static_cast<uint64_t>(ty);
  } else {
    // Original 64-bit packing: uint32 wraparound reproduces pack_key exactly.
    target = pack_key(static_cast<int32_t>(tx), static_cast<int32_t>(ty));
  }
  ngh[c * 25 + off] = binsearch_cell(cell_keys, num_cells, target);
}

// ============================================================================
// Stage 3d: per-cell bounding box of the points actually in the cell, packed
// as float4 [xmin, xmax, ymin, ymax].
//
// Used for conservative distance pruning in the dense-data scans below
// (core pass 2, unite pairs, border assign): if a point (or a whole cell) is
// provably farther than eps from every point of a neighbor cell, the inner
// scan over that cell can be skipped without looking at any of its points.
// Skips only happen when no pair could qualify, so labels are unchanged.
// ============================================================================
__global__ void k_cell_bbox(const float* __restrict__ xs,
                            const float* __restrict__ ys,
                            const int32_t* __restrict__ cell_start,
                            int32_t num_cells, float4* __restrict__ bbox) {
  int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_cells) return;
  int32_t s = cell_start[c], e = cell_start[c + 1];
  float xmin = xs[s], xmax = xs[s], ymin = ys[s], ymax = ys[s];
  for (int32_t i = s + 1; i < e; ++i) {
    xmin = fminf(xmin, xs[i]);
    xmax = fmaxf(xmax, xs[i]);
    ymin = fminf(ymin, ys[i]);
    ymax = fmaxf(ymax, ys[i]);
  }
  bbox[c] = make_float4(xmin, xmax, ymin, ymax);
}

// Squared distance from (x, y) to the box [b.x, b.y] x [b.z, b.w]; zero when
// the point is inside. Uses the same subtract-then-square shape as the
// point-pair distance in the scan kernels, so "box farther than eps" soundly
// implies every point inside the box is farther than eps.
__device__ __forceinline__ float point_box_d2(float x, float y, float4 b) {
  float dx = fmaxf(fmaxf(b.x - x, x - b.y), 0.0f);
  float dy = fmaxf(fmaxf(b.z - y, y - b.w), 0.0f);
  return dx * dx + dy * dy;
}

// Squared distance between two point bounding boxes; zero when they overlap.
__device__ __forceinline__ float box_box_d2(float4 a, float4 b) {
  float dx = fmaxf(fmaxf(b.x - a.y, a.x - b.y), 0.0f);
  float dy = fmaxf(fmaxf(b.z - a.w, a.z - b.w), 0.0f);
  return dx * dx + dy * dy;
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
__device__ const int32_t kCoreScanOrder[25] = {
    12, 6, 7,  8,  11, 13, 16, 17, 18, 0,  1,  2, 3,
    4,  5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24};
__global__ void k_core_pass2(const float* __restrict__ xs,
                             const float* __restrict__ ys,
                             const int32_t* __restrict__ cell_start,
                             const int32_t* __restrict__ point_cell,
                             const int32_t* __restrict__ ngh,
                             const float4* __restrict__ bbox, int32_t n,
                             int32_t min_samples, float eps_sq,
                             uint8_t* __restrict__ is_core) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (is_core[i]) return;
  int32_t c = point_cell[i];
  float xi = xs[i], yi = ys[i];
  const int32_t* row = ngh + c * 25;
  int32_t cnt = 0;
  // Center cell first, then ring 1, then ring 2: nearby cells contribute
  // the most eps-neighbors, so the min_samples early exit usually fires
  // before the mostly-out-of-range ring-2 scans start. The count is
  // commutative, so the visit order cannot change the outcome.
  for (int32_t k = 0; k < 25; ++k) {
    int32_t off = kCoreScanOrder[k];
    int32_t nc = row[off];
    if (nc < 0) continue;
    if (off == 12) {
      cnt += cell_start[nc + 1] - cell_start[nc];
      if (cnt >= min_samples) break;
    } else {
      if (point_box_d2(xi, yi, bbox[nc]) > eps_sq) continue;
      int32_t s = cell_start[nc], e = cell_start[nc + 1];
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

// Read-only walk to root. Used by k_uf_flatten only, where path-halving
// writes would race: a flatten thread t_c writes uf[c] = root as its final
// assignment, while a concurrent flatten thread t_d (with c on d's path)
// can write uf[c] = some_ancestor as path halving inside its find. If t_d's
// write lands after t_c's assignment, uf[c] ends up pointing to a non-root,
// which silently corrupts cluster_id_per_cell. Read-only find sidesteps the
// race: each flatten thread writes only its own uf[c] slot.
__device__ __forceinline__ int32_t uf_find_readonly(const int32_t* uf,
                                                    int32_t x) {
  while (uf[x] != x) x = uf[x];
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

// The 24 neighbor offsets of the 5x5 stencil, grouped by Chebyshev ring.
// Ring 1 (the 8 touching neighbors) is launched first: it alone connects
// nearly every cluster, so by the time ring 2 (the 16 outer offsets, whose
// geometric gap is close to eps and whose scans are the expensive ones)
// launches, almost all of its pairs exit at the already-united check.
__device__ const int32_t kRing1Offsets[8] = {6, 7, 8, 11, 13, 16, 17, 18};
__device__ const int32_t kRing2Offsets[16] = {0,  1,  2,  3,  4,  5,  9, 10,
                                              14, 15, 19, 20, 21, 22, 23, 24};

// Flat thread-per-(cell, ring offset) mapping: full blocks, no per-cell
// block-dispatch overhead (with millions of near-empty cells the old
// one-block-per-cell launch made this kernel dispatch-bound).
__global__ void k_unite_pairs(const float* __restrict__ xs,
                              const float* __restrict__ ys,
                              const int32_t* __restrict__ cell_start,
                              const uint8_t* __restrict__ is_core,
                              const uint8_t* __restrict__ cell_status,
                              const int32_t* __restrict__ ngh,
                              const float4* __restrict__ bbox, int32_t ring,
                              int32_t num_cells, float eps_sq,
                              int32_t* __restrict__ uf) {
  const int32_t* ring_offsets = (ring == 1) ? kRing1Offsets : kRing2Offsets;
  const int32_t ring_count = (ring == 1) ? 8 : 16;
  int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(num_cells) * ring_count) return;
  int32_t c = static_cast<int32_t>(t / ring_count);
  int32_t off = ring_offsets[t % ring_count];
  if (cell_status[c] == 0) return;
  int32_t nc = ngh[c * 25 + off];
  if (nc <= c) return;  // each undirected pair processed once (lower -> higher)
  if (cell_status[nc] == 0) return;

  // If c and nc are already in the same component (via transitive unions by
  // other threads), the inner pair scan is redundant work. Read-only walk:
  // no writes here means no race with other threads' path-halving. A stale
  // view of uf[] can only show *fewer* merges than reality -- never more --
  // so if we conclude "already united" we are correct; if we conclude "not
  // united" we just do the scan, which is the same as before.
  if (uf_find_readonly(uf, c) == uf_find_readonly(uf, nc)) return;

  // Dense-cell pruning. Ring-2 neighbor offsets sit at a geometric gap of up
  // to exactly eps, so a dense cell pair there rarely holds an eps-close pair
  // -- yet without a prune the scan below runs the full O(|c| * |nc|) pass in
  // ONE thread (the 22s-on-blobs-at-10M catastrophe). The box-box check
  // rejects such pairs outright; the per-point check inside the outer loop
  // caps a fruitless scan at O(|c|). Both are conservative: they skip only
  // when no point pair could be within eps, so the partition is unchanged.
  float4 bb_n = bbox[nc];
  if (box_box_d2(bbox[c], bb_n) > eps_sq) return;

  int32_t s_c = cell_start[c], e_c = cell_start[c + 1];
  int32_t s_n = cell_start[nc], e_n = cell_start[nc + 1];
  bool found = false;
  for (int32_t i = s_c; i < e_c && !found; ++i) {
    if (!is_core[i]) continue;
    float xi = xs[i], yi = ys[i];
    if (point_box_d2(xi, yi, bb_n) > eps_sq) continue;
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

// Final compaction so uf[c] points directly to its root. Uses
// uf_find_readonly: see its comment for why path-halving find would race.
__global__ void k_uf_flatten(int32_t* uf, int32_t num_cells) {
  int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_cells) return;
  uf[c] = uf_find_readonly(uf, c);
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
    const float4* __restrict__ bbox,
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
    if (point_box_d2(xi, yi, bbox[nc]) > eps_sq) continue;
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

// Per-stage tensor handles captured by the debug entrypoint. Production
// passes nullptr; the per-stage capture is one predictable branch per
// pipeline (zero measurable perf cost), and a tensor assignment is a
// shared_ptr ref bump, not a copy.
struct DebugIntermediates {
  torch::Tensor keys_sorted;
  torch::Tensor perm;
  torch::Tensor xs;
  torch::Tensor ys;
  torch::Tensor is_core;
  torch::Tensor cell_start;
  torch::Tensor cluster_id_per_cell;
  torch::Tensor uf;
  torch::Tensor sorted_labels_pre_border;
  torch::Tensor sorted_labels_final;
};

// Per-stage GPU-time attribution. Stage boundaries match the production
// pipeline's comment blocks (10 stages). Filled by the impl only when a
// non-null pointer is passed; production path skips event creation entirely.
//
// Caveat: cudaEventElapsedTime captures GPU execution time only. The host
// sync inside the boundary scan (cudaMemcpyAsync + cudaStreamSynchronize to
// read num_cells) shows up as a wallclock cost but NOT in stage_3's GPU
// time. The gap between sum(stage_ms) and total wallclock measures that
// host overhead -- itself a useful attribution signal.
struct StageTimings {
  static constexpr int N = 10;
  float ms[N] = {0};
};
static const char* const kStageNames[StageTimings::N] = {
    "cell_keys",       // stage 1: cell-key computation
    "sort_gather",     // stage 2: sort_by_key + gather xs/ys
    "boundary_scan",   // stage 3: flag + inclusive_scan + extract cell_start
    "build_ngh",       // stage 4: 5x5 neighbor table via binary search
    "core_mark",       // stage 5: pass1 + pass2 + cell_status
    "union_find",      // stage 6: uf_init + unite_pairs + flatten
    "cluster_id",      // stage 7: root_flag + exclusive_scan + cluster_id
    "core_labels",     // stage 8: core-point labels in sort order
    "border_assign",   // stage 9: nearest-core border assignment
    "unpermute",       // stage 10: sort order -> input order
};

}  // namespace

// Shared host pipeline. ``dbg`` and ``timings`` are null on the production
// path -- the per-stage captures collapse to single predictable branches
// with no allocation, no event creation. Pass non-null when you want them.
static torch::Tensor dbscan2d_cuda_impl(torch::Tensor X, double eps,
                                        int64_t min_samples,
                                        DebugIntermediates* dbg,
                                        StageTimings* timings) {
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

  // Stage-boundary events: evs[i] is the *start* of stage (i+1); evs[N]
  // is the end of stage N. elapsed_time(evs[i], evs[i+1]) gives stage i+1
  // GPU time. Only created when ``timings`` was requested.
  cudaEvent_t evs[StageTimings::N + 1] = {nullptr};
  if (timings) {
    for (int i = 0; i <= StageTimings::N; ++i) {
      C10_CUDA_CHECK(cudaEventCreate(&evs[i]));
    }
  }
  auto stage_mark = [&](int i) {
    if (timings) C10_CUDA_CHECK(cudaEventRecord(evs[i], stream));
  };

  // ---------------------------------------------------------------- stage 1
  stage_mark(0);
  // Narrow-key detection (see k_grid_minmax). Only for large inputs: the
  // reduction plus its host sync is fixed latency that small calls would
  // pay without recouping in the sort.
  int32_t min_kx = 0, min_ky = 0;
  int32_t key_shift = 32;
  constexpr int32_t kNarrowMinN = 1 << 19;
  if (n >= kNarrowMinN) {
    auto minmax = torch::empty({4}, opts_i32);
    const int32_t init_vals[4] = {INT32_MAX, INT32_MIN, INT32_MAX, INT32_MIN};
    C10_CUDA_CHECK(cudaMemcpy(minmax.data_ptr<int32_t>(), init_vals,
                              sizeof(init_vals), cudaMemcpyHostToDevice));
    k_grid_minmax<<<grid(n), kBlk, 0, stream>>>(X.data_ptr<float>(), n, inv_r,
                                                minmax.data_ptr<int32_t>());
    int32_t mm[4];
    C10_CUDA_CHECK(cudaMemcpyAsync(mm, minmax.data_ptr<int32_t>(), sizeof(mm),
                                   cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t span_x = static_cast<int64_t>(mm[1]) - mm[0];
    const int64_t span_y = static_cast<int64_t>(mm[3]) - mm[2];
    if (span_x < 65536 && span_y < 65536) {
      min_kx = mm[0];
      min_ky = mm[2];
      key_shift = 16;
    }
  }

  auto keys = torch::empty({n}, opts_i64);
  auto perm = torch::empty({n}, opts_i32);
  k_cell_keys<<<grid(n), kBlk, 0, stream>>>(
      X.data_ptr<float>(), n, inv_r, min_kx, min_ky, key_shift,
      reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
      perm.data_ptr<int32_t>());

  // ---------------------------------------------------------------- stage 2
  // cub radix sort with an explicit bit range: the narrow packing occupies
  // only the low 32 bits, which halves the radix passes vs a blind 64-bit
  // sort. Out-of-place, so the sorted buffers replace the originals.
  stage_mark(1);
  {
    auto keys_out = torch::empty({n}, opts_i64);
    auto perm_out = torch::empty({n}, opts_i32);
    const int end_bit = (key_shift == 16) ? 32 : 64;
    size_t temp_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(keys_out.data_ptr<int64_t>()),
        perm.data_ptr<int32_t>(), perm_out.data_ptr<int32_t>(), n,
        /*begin_bit=*/0, end_bit, stream));
    auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp.data_ptr<uint8_t>(), temp_bytes,
        reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(keys_out.data_ptr<int64_t>()),
        perm.data_ptr<int32_t>(), perm_out.data_ptr<int32_t>(), n,
        /*begin_bit=*/0, end_bit, stream));
    keys = keys_out;
    perm = perm_out;
  }
  if (dbg) { dbg->keys_sorted = keys; dbg->perm = perm; }

  auto xs = torch::empty({n}, X.options());
  auto ys = torch::empty({n}, X.options());
  k_gather_xy<<<grid(n), kBlk, 0, stream>>>(X.data_ptr<float>(),
                                            perm.data_ptr<int32_t>(), n,
                                            xs.data_ptr<float>(),
                                            ys.data_ptr<float>());
  if (dbg) { dbg->xs = xs; dbg->ys = ys; }

  // ---------------------------------------------------------------- stage 3
  stage_mark(2);
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
  if (dbg) { dbg->cell_start = cell_start; }

  // Per-cell point bounding boxes for the conservative distance prunes in
  // core pass 2, unite pairs, and border assign.
  auto bbox = torch::empty({num_cells, 4}, X.options());
  k_cell_bbox<<<grid(num_cells), kBlk, 0, stream>>>(
      xs.data_ptr<float>(), ys.data_ptr<float>(),
      cell_start.data_ptr<int32_t>(), num_cells,
      reinterpret_cast<float4*>(bbox.data_ptr<float>()));

  // ---------------------------------------------------------------- stage 4
  stage_mark(3);
  auto ngh = torch::empty({num_cells, 25}, opts_i32);
  // Flat thread-per-(cell, offset) mapping: full blocks, unlike the previous
  // one-block-per-cell launch that idled 7 of 32 threads and paid a block
  // dispatch per cell (the top stage on uniform data, where cells are many).
  const int64_t ngh_threads = static_cast<int64_t>(num_cells) * 25;
  const int32_t ngh_blocks =
      static_cast<int32_t>((ngh_threads + kBlk - 1) / kBlk);
  k_build_ngh<<<ngh_blocks, kBlk, 0, stream>>>(
      reinterpret_cast<uint64_t*>(cell_keys.data_ptr<int64_t>()), num_cells,
      key_shift, ngh.data_ptr<int32_t>());

  // ---------------------------------------------------------------- stage 5
  stage_mark(4);
  auto is_core = torch::zeros({n}, opts_u8);
  k_core_pass1<<<grid(num_cells), kBlk, 0, stream>>>(
      cell_start.data_ptr<int32_t>(), num_cells, ms,
      is_core.data_ptr<uint8_t>());

  k_core_pass2<<<grid(n), kBlk, 0, stream>>>(
      xs.data_ptr<float>(), ys.data_ptr<float>(),
      cell_start.data_ptr<int32_t>(), point_cell.data_ptr<int32_t>(),
      ngh.data_ptr<int32_t>(),
      reinterpret_cast<const float4*>(bbox.data_ptr<float>()), n, ms, eps_sq,
      is_core.data_ptr<uint8_t>());
  if (dbg) { dbg->is_core = is_core; }

  auto cell_status = torch::empty({num_cells}, opts_u8);
  k_cell_status<<<grid(num_cells), kBlk, 0, stream>>>(
      is_core.data_ptr<uint8_t>(), cell_start.data_ptr<int32_t>(), num_cells,
      cell_status.data_ptr<uint8_t>());

  // ---------------------------------------------------------------- stage 6
  stage_mark(5);
  auto uf = torch::empty({num_cells}, opts_i32);
  k_uf_init<<<grid(num_cells), kBlk, 0, stream>>>(uf.data_ptr<int32_t>(),
                                                  num_cells);

  // Two-phase unite: ring 1 connects nearly everything, the flatten in
  // between makes the already-united check O(1) for ring 2, which then
  // skips most of the expensive near-eps pair scans. The union outcome is
  // order-independent, so the partition (and labels) are unchanged.
  auto launch_unite = [&](int32_t ring, int32_t ring_count) {
    const int64_t unite_threads = static_cast<int64_t>(num_cells) * ring_count;
    const int32_t unite_blocks =
        static_cast<int32_t>((unite_threads + kBlk - 1) / kBlk);
    k_unite_pairs<<<unite_blocks, kBlk, 0, stream>>>(
        xs.data_ptr<float>(), ys.data_ptr<float>(),
        cell_start.data_ptr<int32_t>(), is_core.data_ptr<uint8_t>(),
        cell_status.data_ptr<uint8_t>(), ngh.data_ptr<int32_t>(),
        reinterpret_cast<const float4*>(bbox.data_ptr<float>()), ring,
        num_cells, eps_sq, uf.data_ptr<int32_t>());
    k_uf_flatten<<<grid(num_cells), kBlk, 0, stream>>>(uf.data_ptr<int32_t>(),
                                                       num_cells);
  };
  launch_unite(1, 8);
  launch_unite(2, 16);
  if (dbg) { dbg->uf = uf; }

  // ---------------------------------------------------------------- stage 7
  stage_mark(6);
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
  if (dbg) { dbg->cluster_id_per_cell = cluster_id; }

  // ---------------------------------------------------------------- stage 8
  stage_mark(7);
  auto sorted_labels = torch::empty({n}, opts_i32);
  k_core_labels<<<grid(n), kBlk, 0, stream>>>(
      is_core.data_ptr<uint8_t>(), point_cell.data_ptr<int32_t>(),
      cluster_id.data_ptr<int32_t>(), n, sorted_labels.data_ptr<int32_t>());
  if (dbg) { dbg->sorted_labels_pre_border = sorted_labels; }

  // ---------------------------------------------------------------- stage 9
  stage_mark(8);
  auto sorted_labels_final = torch::empty({n}, opts_i32);
  k_border_assign<<<grid(n), kBlk, 0, stream>>>(
      xs.data_ptr<float>(), ys.data_ptr<float>(), is_core.data_ptr<uint8_t>(),
      cell_start.data_ptr<int32_t>(), point_cell.data_ptr<int32_t>(),
      ngh.data_ptr<int32_t>(), cell_status.data_ptr<uint8_t>(),
      reinterpret_cast<const float4*>(bbox.data_ptr<float>()),
      perm.data_ptr<int32_t>(), sorted_labels.data_ptr<int32_t>(), n, eps_sq,
      sorted_labels_final.data_ptr<int32_t>());
  if (dbg) { dbg->sorted_labels_final = sorted_labels_final; }

  // ---------------------------------------------------------------- stage 10
  stage_mark(9);
  auto labels = torch::full({n}, kNoise, opts_i32);
  k_unpermute<<<grid(n), kBlk, 0, stream>>>(
      perm.data_ptr<int32_t>(), sorted_labels_final.data_ptr<int32_t>(), n,
      labels.data_ptr<int32_t>());
  stage_mark(10);

  C10_CUDA_CHECK(cudaGetLastError());

  if (timings) {
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < StageTimings::N; ++i) {
      C10_CUDA_CHECK(cudaEventElapsedTime(&timings->ms[i], evs[i], evs[i + 1]));
    }
    for (int i = 0; i <= StageTimings::N; ++i) {
      C10_CUDA_CHECK(cudaEventDestroy(evs[i]));
    }
  }
  return labels;
}

torch::Tensor dbscan2d_cuda(torch::Tensor X, double eps, int64_t min_samples) {
  return dbscan2d_cuda_impl(std::move(X), eps, min_samples,
                            /*dbg=*/nullptr, /*timings=*/nullptr);
}

// Profile entrypoint: runs the production pipeline, returns labels plus a
// per-stage GPU-time map. Used by the eval loop's --profile mode for
// attribution-driven optimization. Stage names match kStageNames.
std::pair<torch::Tensor, std::map<std::string, float>>
dbscan2d_cuda_profile(torch::Tensor X, double eps, int64_t min_samples) {
  StageTimings timings;
  auto labels = dbscan2d_cuda_impl(std::move(X), eps, min_samples,
                                   /*dbg=*/nullptr, &timings);
  std::map<std::string, float> out;
  for (int i = 0; i < StageTimings::N; ++i) {
    out[kStageNames[i]] = timings.ms[i];
  }
  return {labels, out};
}

// ============================================================================
// Debug entrypoint: runs the same pipeline as ``dbscan2d_cuda`` and returns
// the per-stage intermediate tensors so a Python script can diff two runs
// and find the first divergent stage. Not part of the public API.
//
// Return order (also documented in evals/investigate_determinism.py):
//   0  labels                  (n,)        int32   final output
//   1  keys_sorted             (n,)        int64   uint64-packed cell keys (sorted)
//   2  perm                    (n,)        int32   sort permutation (sorted -> original)
//   3  xs                      (n,)        float32 x-coords in sorted order
//   4  ys                      (n,)        float32 y-coords in sorted order
//   5  is_core                 (n,)        uint8   core flag in sorted order
//   6  cell_start              (num_cells+1,) int32
//   7  cluster_id_per_cell     (num_cells,)   int32
//   8  uf                      (num_cells,)   int32   path-compressed roots
//   9  sorted_labels_pre_bord  (n,)        int32   core labels, before border-assign
//   10 sorted_labels_final     (n,)        int32   after border-assign, before unpermute
// ============================================================================
std::vector<torch::Tensor> dbscan2d_cuda_debug(torch::Tensor X, double eps,
                                               int64_t min_samples) {
  TORCH_CHECK(X.dim() == 2 && X.size(0) > 0, "debug variant requires n > 0");
  DebugIntermediates dbg;
  auto labels = dbscan2d_cuda_impl(std::move(X), eps, min_samples, &dbg,
                                   /*timings=*/nullptr);
  return {labels,           dbg.keys_sorted,         dbg.perm,
          dbg.xs,           dbg.ys,                  dbg.is_core,
          dbg.cell_start,   dbg.cluster_id_per_cell, dbg.uf,
          dbg.sorted_labels_pre_border,              dbg.sorted_labels_final};
}
