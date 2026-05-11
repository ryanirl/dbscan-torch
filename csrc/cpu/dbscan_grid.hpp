// dbscan_grid.hpp -- header-only grid-hashed 2D DBSCAN, parallel CPU.
//
// Cell side r = eps/sqrt(2) so:
//   - any two points in the same cell are within `eps` (max in-cell distance
//     is r*sqrt(2) = eps), so same-cell pairs need no distance check.
//   - all candidate neighbors of a point lie in a 5x5 cell window.
//
// Layout: points are sorted by cell key once, so each cell is a contiguous
// slice of `xs`, `ys`, `labels`. A 25-entry neighbor-cell table is
// precomputed per cell, eliminating hashmap traffic from the hot loop.
//
// Parallelism: at::parallel_for over the embarrassingly-parallel stages
// (cell-key compute, reorder, both core-marking passes, cell status, UF
// unions, core labels, border, unpermute). Cell-level union-find uses
// std::atomic<int32_t> with lock-free CAS unite + path-halving find.
// Sorting is delegated to at::sort, which is parallel under torch's existing
// threading layer -- no extra build deps. The cell-boundary scan and the
// root-to-cluster id assignment stay sequential (small relative to N).
//
// Thread count follows torch.get_num_threads() / OMP_NUM_THREADS, like any
// other torch op.

#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dbscan_grid {

constexpr int32_t kNoise = -1;

inline uint64_t cell_key(int32_t kx, int32_t ky) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(kx)) << 32) |
         static_cast<uint32_t>(ky);
}

// Inverse of cell_key: extracts (kx, ky) from a packed uint64.
inline std::pair<int32_t, int32_t> unpack_cell_key(uint64_t k) {
  return {static_cast<int32_t>(static_cast<uint32_t>(k >> 32)),
          static_cast<int32_t>(static_cast<uint32_t>(k))};
}

struct Result {
  std::vector<int32_t> labels;  // -1 = noise, otherwise [0, n_clusters)
  int32_t n_clusters = 0;
};

// Lock-free atomic union-find (ECL-CC pattern). find() does path halving,
// relaxed loads/stores are safe because writes only redirect to valid
// ancestors, which only ever flow toward smaller-id roots. unite() loops on
// an atomic CAS that hooks the larger root onto the smaller.
struct AtomicUF {
  std::vector<std::atomic<int32_t>> uf;

  explicit AtomicUF(int32_t n) : uf(n) {
    at::parallel_for(0, n, 1024, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        uf[i].store(static_cast<int32_t>(i), std::memory_order_relaxed);
      }
    });
  }

  int32_t find(int32_t x) {
    int32_t p = uf[x].load(std::memory_order_relaxed);
    while (p != x) {
      int32_t gp = uf[p].load(std::memory_order_relaxed);
      uf[x].store(gp, std::memory_order_relaxed);  // path halving
      x = p;
      p = gp;
    }
    return x;
  }

  void unite(int32_t a, int32_t b) {
    while (true) {
      a = find(a);
      b = find(b);
      if (a == b) return;
      int32_t hi = std::max(a, b), lo = std::min(a, b);
      int32_t expected = hi;
      if (uf[hi].compare_exchange_strong(expected, lo,
                                         std::memory_order_relaxed)) return;
    }
  }
};

inline Result dbscan2d(const float* xs_in, const float* ys_in, int32_t n,
                       float eps, int32_t min_samples) {
  Result out;
  out.labels.assign(static_cast<size_t>(n), kNoise);
  if (n == 0 || min_samples < 1) return out;

  const float inv_r = static_cast<float>(std::sqrt(2.0)) / eps;
  const float eps_sq = eps * eps;

  // Per-point packed cell key. Cell coords get unpacked from the (sorted) key
  // at cell boundaries below -- no need to keep separate kxs/kys.
  std::vector<uint64_t> keys(n);
  at::parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      int32_t kx = static_cast<int32_t>(std::floor(xs_in[i] * inv_r));
      int32_t ky = static_cast<int32_t>(std::floor(ys_in[i] * inv_r));
      keys[i] = cell_key(kx, ky);
    }
  });

  // Sort by key via at::sort. We need BOTH the sorted keys (for the boundary
  // walk -- sequential reads, no gather) and the argsort permutation (for the
  // xs/ys reorder).
  std::vector<int32_t> perm(n);
  at::Tensor sorted_keys_t;
  {
    auto keys_t = at::from_blob(reinterpret_cast<int64_t*>(keys.data()),
                                {n}, at::kLong);
    auto sort_result = at::sort(keys_t, /*dim=*/-1, /*descending=*/false);
    sorted_keys_t = std::get<0>(sort_result);
    auto perm_t = std::get<1>(sort_result);
    const int64_t* p_i64 = perm_t.data_ptr<int64_t>();
    at::parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        perm[i] = static_cast<int32_t>(p_i64[i]);
      }
    });
  }
  std::vector<uint64_t>().swap(keys);  // original-order keys no longer needed
  const uint64_t* sorted_keys =
      reinterpret_cast<const uint64_t*>(sorted_keys_t.data_ptr<int64_t>());

  // Reorder xs/ys into sort order. Cell coords are rederived from sorted_keys
  // at the boundary walk below, so we don't gather them here.
  std::vector<float> xs(n), ys(n);
  at::parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      int32_t p = perm[i];
      xs[i] = xs_in[p];
      ys[i] = ys_in[p];
    }
  });

  // Cell boundaries -- sequential walk over sorted_keys (cache-friendly).
  std::vector<int32_t> cell_start;
  std::vector<int32_t> cell_kx, cell_ky;
  cell_start.reserve(n / 64 + 1);
  cell_start.push_back(0);
  {
    auto [kx0, ky0] = unpack_cell_key(sorted_keys[0]);
    cell_kx.push_back(kx0);
    cell_ky.push_back(ky0);
  }
  uint64_t prev_key = sorted_keys[0];
  for (int32_t i = 1; i < n; ++i) {
    uint64_t k = sorted_keys[i];
    if (k != prev_key) {
      cell_start.push_back(i);
      auto [kx, ky] = unpack_cell_key(k);
      cell_kx.push_back(kx);
      cell_ky.push_back(ky);
      prev_key = k;
    }
  }
  cell_start.push_back(n);
  const int32_t num_cells = static_cast<int32_t>(cell_kx.size());

  // Hashmap: cell key -> cell index. Sequential build (small).
  std::unordered_map<uint64_t, int32_t> cell_lookup;
  cell_lookup.reserve(static_cast<size_t>(num_cells * 2));
  for (int32_t c = 0; c < num_cells; ++c) {
    cell_lookup.emplace(cell_key(cell_kx[c], cell_ky[c]), c);
  }

  // 5x5 neighbor table per cell. Parallel.
  std::vector<int32_t> ngh(static_cast<size_t>(num_cells) * 25, -1);
  at::parallel_for(0, num_cells, 256, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; ++c) {
      const int32_t kx = cell_kx[c], ky = cell_ky[c];
      int32_t* row = ngh.data() + size_t(c) * 25;
      for (int32_t dx = -2; dx <= 2; ++dx) {
        for (int32_t dy = -2; dy <= 2; ++dy) {
          auto it = cell_lookup.find(cell_key(kx + dx, ky + dy));
          row[(dx + 2) * 5 + (dy + 2)] =
              (it == cell_lookup.end()) ? -1 : it->second;
        }
      }
    }
  });
  std::unordered_map<uint64_t, int32_t>().swap(cell_lookup);

  // Per-(sorted-)point cell index. Parallel via cell ranges.
  std::vector<int32_t> point_cell(n);
  at::parallel_for(0, num_cells, 64, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; ++c) {
      for (int32_t i = cell_start[c]; i < cell_start[c + 1]; ++i) {
        point_cell[i] = static_cast<int32_t>(c);
      }
    }
  });

  // Mark core points.
  std::vector<uint8_t> is_core(n, 0);

  // Pass 1: full-core cells. Parallel over cells.
  at::parallel_for(0, num_cells, 256, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; ++c) {
      if (cell_start[c + 1] - cell_start[c] >= min_samples) {
        for (int32_t i = cell_start[c]; i < cell_start[c + 1]; ++i) {
          is_core[i] = 1;
        }
      }
    }
  });

  // Pass 2: per-point sweep over the 5x5 window. Parallel, one of the hot loops.
  // OpenMP dynamic schedule: per-point work is highly uneven (early-exit at
  // min_samples), so static block-static splitting leaves threads idle.
#pragma omp parallel for schedule(dynamic, 256)
  for (int64_t i = 0; i < n; ++i) {
    if (is_core[i]) continue;
    const int32_t c = point_cell[i];
    const float xi = xs[i], yi = ys[i];
    const int32_t* row = ngh.data() + size_t(c) * 25;
    int32_t cnt = 0;
    for (int32_t off = 0; off < 25; ++off) {
      const int32_t nc = row[off];
      if (nc < 0) continue;
      const int32_t s = cell_start[nc], en = cell_start[nc + 1];
      if (off == 12) {
        cnt += (en - s);
        if (cnt >= min_samples) break;
      } else {
        for (int32_t j = s; j < en; ++j) {
          const float ex = xs[j] - xi, ey = ys[j] - yi;
          if (ex * ex + ey * ey <= eps_sq) ++cnt;
        }
        if (cnt >= min_samples) break;
      }
    }
    if (cnt >= min_samples) is_core[i] = 1;
  }

  // Per-cell core summary: 0 = no core, 1 = mixed, 2 = fully core. Parallel.
  std::vector<uint8_t> cell_status(num_cells, 0);
  at::parallel_for(0, num_cells, 256, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; ++c) {
      const int32_t s = cell_start[c], en = cell_start[c + 1];
      bool any = false, all = (en > s);
      for (int32_t j = s; j < en; ++j) {
        if (is_core[j]) any = true;
        else all = false;
      }
      cell_status[c] = !any ? 0 : (all ? 2 : 1);
    }
  });

  // Cell-level union-find. Atomic-CAS unite. Parallel over cells.
  // OpenMP dynamic schedule: per-cell work varies (cell point counts +
  // early-exit on first eps-close core pair).
  AtomicUF uf(num_cells);
#pragma omp parallel for schedule(dynamic, 32)
  for (int64_t c = 0; c < num_cells; ++c) {
    if (cell_status[c] == 0) continue;
    const int32_t* row = ngh.data() + size_t(c) * 25;
    const int32_t s_c = cell_start[c], e_c = cell_start[c + 1];
    for (int32_t off = 0; off < 25; ++off) {
      if (off == 12) continue;
      const int32_t nc = row[off];
      if (nc <= static_cast<int32_t>(c)) continue;
      if (cell_status[nc] == 0) continue;
      if (uf.find(static_cast<int32_t>(c)) == uf.find(nc)) continue;

      const int32_t s_n = cell_start[nc], e_n = cell_start[nc + 1];
      bool found = false;
      for (int32_t i = s_c; i < e_c && !found; ++i) {
        if (!is_core[i]) continue;
        const float xi = xs[i], yi = ys[i];
        for (int32_t j = s_n; j < e_n; ++j) {
          if (!is_core[j]) continue;
          const float ex = xs[j] - xi, ey = ys[j] - yi;
          if (ex * ex + ey * ey <= eps_sq) {
            found = true;
            break;
          }
        }
      }
      if (found) uf.unite(static_cast<int32_t>(c), nc);
    }
  }

  // Renumber roots to dense cluster ids in cell order. Sequential.
  std::vector<int32_t> root_to_cluster(num_cells, kNoise);
  int32_t cid = 0;
  for (int32_t c = 0; c < num_cells; ++c) {
    if (cell_status[c] == 0) continue;
    const int32_t r = uf.find(c);
    if (root_to_cluster[r] == kNoise) root_to_cluster[r] = cid++;
  }

  // Assign cluster id to each core point. Parallel over cells.
  std::vector<int32_t> sorted_labels(n, kNoise);
  at::parallel_for(0, num_cells, 256, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; ++c) {
      if (cell_status[c] == 0) continue;
      const int32_t lab = root_to_cluster[uf.find(static_cast<int32_t>(c))];
      for (int32_t j = cell_start[c]; j < cell_start[c + 1]; ++j) {
        if (is_core[j]) sorted_labels[j] = lab;
      }
    }
  });

  // Border assignment: each non-core within eps of some core takes the
  // cluster of the nearest such core, otherwise it stays noise. Parallel
  // over points.
  at::parallel_for(0, n, 1024, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      if (is_core[i]) continue;
      const int32_t c = point_cell[i];
      const int32_t* row = ngh.data() + size_t(c) * 25;
      const float xi = xs[i], yi = ys[i];
      float best_d2 = eps_sq + 1.0f;  // strict-better criterion below
      int32_t best_lab = kNoise;
      for (int32_t off = 0; off < 25; ++off) {
        const int32_t nc = row[off];
        if (nc < 0 || cell_status[nc] == 0) continue;
        const int32_t s = cell_start[nc], en = cell_start[nc + 1];
        for (int32_t j = s; j < en; ++j) {
          if (!is_core[j]) continue;
          const float ex = xs[j] - xi, ey = ys[j] - yi;
          const float d2 = ex * ex + ey * ey;
          if (d2 <= eps_sq && d2 < best_d2) {
            best_d2 = d2;
            best_lab = sorted_labels[j];
          }
        }
      }
      sorted_labels[i] = best_lab;
    }
  });

  // Unpermute back to input order. Parallel.
  at::parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      out.labels[perm[i]] = sorted_labels[i];
    }
  });
  out.n_clusters = cid;
  return out;
}

}  // namespace dbscan_grid
