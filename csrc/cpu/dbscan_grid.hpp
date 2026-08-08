// dbscan_grid.hpp -- header-only grid-hashed 2D DBSCAN, parallel CPU.
//
// Cell side r = eps/sqrt(2) so:
//   - any two points in the same cell are within `eps` (max in-cell distance
//     is r*sqrt(2) = eps), so same-cell pairs need no distance check.
//   - all candidate neighbors of a point lie in a 5x5 cell window.
//
// Layout: points are sorted by cell key once, so each cell is a contiguous
// slice of `xs`, `ys`, `labels`. A 25-entry neighbor-cell table is
// precomputed per cell by a column-pair merge-join over the sorted cells,
// keeping hashmap and binary-search traffic out of the hot loops.
//
// Keys: when a large input's occupied grid spans fewer than 2^16 cells per
// axis (any realistic dataset), coords pack as 16+16-bit offsets into a
// 32-bit key sorted by a 2-pass parallel radix sort; otherwise (and for
// small inputs) 64-bit sign-biased keys are sorted via at::sort.
//
// Parallelism: at::parallel_for over the embarrassingly-parallel stages
// (cell-key compute, reorder, boundary scan, both core-marking passes, cell
// status, UF unions, core labels, border, unpermute). Cell-level union-find
// uses std::atomic<int32_t> with lock-free CAS unite + path-halving find.
// Chunked stages use a fixed chunk grid (independent of thread count), so
// output is bit-identical under any parallelism. Only the root-to-cluster
// id assignment stays sequential (small relative to N).
//
// Thread count follows torch.get_num_threads() / OMP_NUM_THREADS, like any
// other torch op.

#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace dbscan_grid {

constexpr int32_t kNoise = -1;

inline uint64_t cell_key(int32_t kx, int32_t ky) {
  // The sign bit is XORed into both coords so unsigned key order equals
  // signed (kx, ky) lexicographic order. The neighbor-table build walks
  // contiguous ky-windows in the sorted key array; without the bias a
  // window crossing ky = 0 splits into two runs and neighbors near the
  // axes would be silently missed.
  return (static_cast<uint64_t>(static_cast<uint32_t>(kx) ^ 0x80000000u) << 32) |
         (static_cast<uint32_t>(ky) ^ 0x80000000u);
}

// Inverse of cell_key: extracts (kx, ky) from a packed uint64.
inline std::pair<int32_t, int32_t> unpack_cell_key(uint64_t k) {
  return {static_cast<int32_t>(static_cast<uint32_t>(k >> 32) ^ 0x80000000u),
          static_cast<int32_t>(static_cast<uint32_t>(k) ^ 0x80000000u)};
}

struct Result {
  std::vector<int32_t> labels;  // -1 = noise, otherwise [0, n_clusters)
  int32_t n_clusters = 0;
};

// Per-stage wallclock attribution. Stage names mirror the CUDA path's
// kStageNames so cross-device profiles are directly comparable. Filled
// only when ``dbscan2d`` is called with a non-null pointer; production
// passes nullptr and the per-stage capture compiles down to a single
// predictable branch (no clock reads, no allocation).
struct Timings {
  static constexpr int N = 10;
  double ms[N] = {0};
};
static const char* const kStageNames[Timings::N] = {
    "cell_keys",       // stage 1
    "sort_gather",     // stage 2
    "boundary_scan",   // stage 3
    "build_ngh",       // stage 4
    "core_mark",       // stage 5
    "union_find",      // stage 6
    "cluster_id",      // stage 7
    "core_labels",     // stage 8
    "border_assign",   // stage 9
    "unpermute",       // stage 10
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

// Stable 2x16-bit LSD radix sort of (key, value) pairs for 32-bit keys.
// Parallel and deterministic: a fixed chunk grid (independent of thread
// count), per-chunk histograms, bucket-major prefix, chunk-stable scatter.
// Two passes of sequential memory traffic, which beats a comparison sort on
// 64-bit keys by a wide margin at the sizes where it is enabled.
inline void radix_sort_pairs32(std::vector<uint32_t>& keys,
                               std::vector<int32_t>& vals) {
  const int64_t n = static_cast<int64_t>(keys.size());
  constexpr int64_t kChunks = 64;
  constexpr int64_t kBuckets = 65536;
  std::vector<uint32_t> keys_tmp(n);
  std::vector<int32_t> vals_tmp(n);
  std::vector<int32_t> hist(kChunks * kBuckets);
  auto chunk_lo = [&](int64_t chunk) { return n * chunk / kChunks; };

  for (int pass = 0; pass < 2; ++pass) {
    const int shift = pass * 16;

    at::parallel_for(0, kChunks, 1, [&](int64_t b, int64_t e) {
      for (int64_t chunk = b; chunk < e; ++chunk) {
        int32_t* h = hist.data() + chunk * kBuckets;
        std::fill(h, h + kBuckets, 0);
        for (int64_t i = chunk_lo(chunk); i < chunk_lo(chunk + 1); ++i) {
          ++h[(keys[i] >> shift) & 0xFFFF];
        }
      }
    });

    // Per-bucket totals, then rewrite hist in place to per-(chunk, bucket)
    // starting offsets: bucket-major, chunk-minor, so equal keys keep their
    // chunk order (stability).
    std::vector<int64_t> bucket_total(kBuckets);
    at::parallel_for(0, kBuckets, 1024, [&](int64_t b, int64_t e) {
      for (int64_t bucket = b; bucket < e; ++bucket) {
        int64_t total = 0;
        for (int64_t chunk = 0; chunk < kChunks; ++chunk) {
          total += hist[chunk * kBuckets + bucket];
        }
        bucket_total[bucket] = total;
      }
    });
    int64_t running = 0;
    std::vector<int64_t> bucket_base(kBuckets);
    for (int64_t bucket = 0; bucket < kBuckets; ++bucket) {
      bucket_base[bucket] = running;
      running += bucket_total[bucket];
    }
    at::parallel_for(0, kBuckets, 1024, [&](int64_t b, int64_t e) {
      for (int64_t bucket = b; bucket < e; ++bucket) {
        int64_t offset = bucket_base[bucket];
        for (int64_t chunk = 0; chunk < kChunks; ++chunk) {
          const int32_t count = hist[chunk * kBuckets + bucket];
          hist[chunk * kBuckets + bucket] = static_cast<int32_t>(offset - bucket_base[bucket]);
          offset += count;
        }
      }
    });

    at::parallel_for(0, kChunks, 1, [&](int64_t b, int64_t e) {
      for (int64_t chunk = b; chunk < e; ++chunk) {
        int32_t* h = hist.data() + chunk * kBuckets;
        for (int64_t i = chunk_lo(chunk); i < chunk_lo(chunk + 1); ++i) {
          const int64_t bucket = (keys[i] >> shift) & 0xFFFF;
          const int64_t dst = bucket_base[bucket] + h[bucket]++;
          keys_tmp[dst] = keys[i];
          vals_tmp[dst] = vals[i];
        }
      }
    });

    keys.swap(keys_tmp);
    vals.swap(vals_tmp);
  }
}

// Core pass 2 visit order: center cell, then ring 1, then ring 2. Nearby
// cells contribute the most eps-neighbors, so counting them first lets the
// min_samples early exit fire before the mostly-out-of-range ring-2 scans.
// The count is commutative, so the order cannot change the outcome.
static const int32_t kCoreScanOrder[25] = {
    12, 6, 7,  8,  11, 13, 16, 17, 18, 0,  1,  2, 3,
    4,  5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24};

// Squared distance from (x, y) to the box [box[0], box[1]] x [box[2], box[3]]
// (a per-cell point bounding box); zero when the point is inside. Same
// subtract-then-square shape as the point-pair distance in the scans, so
// "box farther than eps" soundly implies every point inside is farther too.
inline float point_box_d2(float x, float y, const float* box) {
  const float dx = std::max({box[0] - x, x - box[1], 0.0f});
  const float dy = std::max({box[2] - y, y - box[3], 0.0f});
  return dx * dx + dy * dy;
}

// Squared distance between two point bounding boxes; zero when they overlap.
inline float box_box_d2(const float* a, const float* b) {
  const float dx = std::max({b[0] - a[1], a[0] - b[1], 0.0f});
  const float dy = std::max({b[2] - a[3], a[2] - b[3], 0.0f});
  return dx * dx + dy * dy;
}

inline Result dbscan2d(const float* xs_in, const float* ys_in, int32_t n,
                       float eps, int32_t min_samples,
                       Timings* timings = nullptr) {
  Result out;
  out.labels.assign(static_cast<size_t>(n), kNoise);
  if (n == 0 || min_samples < 1) return out;

  using clock = std::chrono::steady_clock;
  auto stage_t0 = clock::now();
  auto stage_end = [&](int i) {
    if (timings) {
      auto t1 = clock::now();
      timings->ms[i] = std::chrono::duration<double, std::milli>(t1 - stage_t0).count();
      stage_t0 = t1;
    }
  };

  const float inv_r = static_cast<float>(std::sqrt(2.0)) / eps;
  const float eps_sq = eps * eps;

  // ---------------------------------------------------------------- stage 1
  // Narrow-key detection: when the occupied grid spans fewer than 2^16
  // cells per axis (any realistic dataset), keys pack into 32 bits as
  // offsets from the grid minimum, and stage 2 uses the 2-pass radix sort
  // instead of at::sort on 64-bit keys. Only checked for large inputs; the
  // extra pass over the input would not pay below that.
  constexpr int64_t kNarrowMinN = 1 << 22;
  bool narrow = false;
  int32_t min_kx = 0, min_ky = 0;
  if (n >= kNarrowMinN) {
    constexpr int64_t kMinMaxChunks = 64;
    int32_t mins[4 * kMinMaxChunks];  // per-chunk min_kx, max_kx, min_ky, max_ky
    at::parallel_for(0, kMinMaxChunks, 1, [&](int64_t b, int64_t e) {
      for (int64_t chunk = b; chunk < e; ++chunk) {
        int32_t lo_x = INT32_MAX, hi_x = INT32_MIN;
        int32_t lo_y = INT32_MAX, hi_y = INT32_MIN;
        for (int64_t i = n * chunk / kMinMaxChunks;
             i < n * (chunk + 1) / kMinMaxChunks; ++i) {
          const int32_t kx = static_cast<int32_t>(std::floor(xs_in[i] * inv_r));
          const int32_t ky = static_cast<int32_t>(std::floor(ys_in[i] * inv_r));
          lo_x = std::min(lo_x, kx);
          hi_x = std::max(hi_x, kx);
          lo_y = std::min(lo_y, ky);
          hi_y = std::max(hi_y, ky);
        }
        mins[4 * chunk + 0] = lo_x;
        mins[4 * chunk + 1] = hi_x;
        mins[4 * chunk + 2] = lo_y;
        mins[4 * chunk + 3] = hi_y;
      }
    });
    int32_t lo_x = INT32_MAX, hi_x = INT32_MIN;
    int32_t lo_y = INT32_MAX, hi_y = INT32_MIN;
    for (int64_t chunk = 0; chunk < kMinMaxChunks; ++chunk) {
      lo_x = std::min(lo_x, mins[4 * chunk + 0]);
      hi_x = std::max(hi_x, mins[4 * chunk + 1]);
      lo_y = std::min(lo_y, mins[4 * chunk + 2]);
      hi_y = std::max(hi_y, mins[4 * chunk + 3]);
    }
    const int64_t span_x = static_cast<int64_t>(hi_x) - lo_x;
    const int64_t span_y = static_cast<int64_t>(hi_y) - lo_y;
    if (span_x < 65536 && span_y < 65536) {
      narrow = true;
      min_kx = lo_x;
      min_ky = lo_y;
    }
  }

  // Per-point packed cell key (width per the detection above). Cell coords
  // get unpacked from the (sorted) key at cell boundaries below.
  std::vector<uint64_t> keys64;
  std::vector<uint32_t> keys32;
  if (narrow) {
    keys32.resize(n);
    at::parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        const int32_t kx = static_cast<int32_t>(std::floor(xs_in[i] * inv_r));
        const int32_t ky = static_cast<int32_t>(std::floor(ys_in[i] * inv_r));
        keys32[i] = (static_cast<uint32_t>(kx - min_kx) << 16) |
                    static_cast<uint32_t>(ky - min_ky);
      }
    });
  } else {
    keys64.resize(n);
    at::parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        const int32_t kx = static_cast<int32_t>(std::floor(xs_in[i] * inv_r));
        const int32_t ky = static_cast<int32_t>(std::floor(ys_in[i] * inv_r));
        keys64[i] = cell_key(kx, ky);
      }
    });
  }
  stage_end(0);

  // ---------------------------------------------------------------- stage 2
  // Sort by key. Narrow path: 2-pass radix on (key32, index), which also
  // yields the int32 permutation directly. Fallback: at::sort on int64 keys
  // plus an index-width conversion. Either way we keep BOTH the sorted keys
  // (for the boundary walk) and the permutation (for the xs/ys reorder).
  std::vector<int32_t> perm(n);
  at::Tensor sorted_keys_t;
  const uint64_t* sorted_keys = nullptr;
  if (narrow) {
    at::parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        perm[i] = static_cast<int32_t>(i);
      }
    });
    radix_sort_pairs32(keys32, perm);
  } else {
    auto keys_t = at::from_blob(reinterpret_cast<int64_t*>(keys64.data()),
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
    std::vector<uint64_t>().swap(keys64);  // original-order keys done
    sorted_keys =
        reinterpret_cast<const uint64_t*>(sorted_keys_t.data_ptr<int64_t>());
  }

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
  stage_end(1);

  // ---------------------------------------------------------------- stage 3
  // Cell boundaries. Deterministic chunked two-pass scan: count boundary
  // flags per fixed chunk, prefix the chunk counts sequentially (tiny),
  // then fill. The chunk grid depends only on n -- never on the thread
  // count -- so the output is bit-identical under any parallelism. Generic
  // over the two key widths; on the narrow path cell_kx/cell_ky hold the
  // non-negative OFFSET coords (downstream only ever uses differences and
  // ordering, and non-negative values keep the column search's unsigned
  // comparator valid).
  const int64_t num_chunks = std::min<int64_t>(256, std::max<int64_t>(1, n));
  auto chunk_lo = [&](int64_t chunk) {
    return 1 + (static_cast<int64_t>(n) - 1) * chunk / num_chunks;
  };
  int32_t num_cells = 0;
  std::vector<int32_t> cell_start, cell_kx, cell_ky;
  auto scan_boundaries = [&](const auto* skeys, auto unpack_fn) {
    std::vector<int64_t> chunk_offset(num_chunks + 1, 0);
    at::parallel_for(0, num_chunks, 1, [&](int64_t b, int64_t e) {
      for (int64_t chunk = b; chunk < e; ++chunk) {
        int64_t count = 0;
        for (int64_t i = chunk_lo(chunk); i < chunk_lo(chunk + 1); ++i) {
          count += (skeys[i] != skeys[i - 1]);
        }
        chunk_offset[chunk + 1] = count;
      }
    });
    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
      chunk_offset[chunk + 1] += chunk_offset[chunk];
    }

    num_cells = static_cast<int32_t>(chunk_offset[num_chunks]) + 1;
    cell_start.assign(static_cast<size_t>(num_cells) + 1, 0);
    cell_kx.resize(num_cells);
    cell_ky.resize(num_cells);
    cell_start[0] = 0;
    cell_start[num_cells] = n;
    {
      auto [kx0, ky0] = unpack_fn(skeys[0]);
      cell_kx[0] = kx0;
      cell_ky[0] = ky0;
    }
    at::parallel_for(0, num_chunks, 1, [&](int64_t b, int64_t e) {
      for (int64_t chunk = b; chunk < e; ++chunk) {
        int64_t c = chunk_offset[chunk] + 1;
        for (int64_t i = chunk_lo(chunk); i < chunk_lo(chunk + 1); ++i) {
          if (skeys[i] != skeys[i - 1]) {
            cell_start[c] = static_cast<int32_t>(i);
            auto [kx, ky] = unpack_fn(skeys[i]);
            cell_kx[c] = kx;
            cell_ky[c] = ky;
            ++c;
          }
        }
      }
    });
  };
  if (narrow) {
    auto unpack32 = [](uint32_t k) {
      return std::pair<int32_t, int32_t>(static_cast<int32_t>(k >> 16),
                                         static_cast<int32_t>(k & 0xFFFF));
    };
    scan_boundaries(keys32.data(), unpack32);
    std::vector<uint32_t>().swap(keys32);  // sorted keys no longer needed
  } else {
    auto unpack64 = [](uint64_t k) { return unpack_cell_key(k); };
    scan_boundaries(sorted_keys, unpack64);
  }

  stage_end(2);

  // ---------------------------------------------------------------- stage 4
  // 5x5 neighbor table via column-pair merge-join. Cells sharing a kx form
  // a contiguous, ky-sorted run ("column"); after the signed-int64 sort of
  // sign-biased keys the columns themselves are ordered by uint32(kx). For
  // each (column, dx) pair a single two-pointer walk over the two ky-sorted
  // runs fills every cell's row in O(cells) sequential access -- no
  // per-cell hashmap probes or binary searches (whose scattered loads
  // dominated this stage).
  std::vector<int32_t> col_of_kx_start;
  {
    std::vector<int64_t> col_chunk(num_chunks + 1, 0);
    const int64_t cell_chunks = std::min<int64_t>(num_chunks, num_cells);
    auto ccl = [&](int64_t chunk) {
      return 1 + (static_cast<int64_t>(num_cells) - 1) * chunk / cell_chunks;
    };
    at::parallel_for(0, cell_chunks, 1, [&](int64_t b, int64_t e) {
      for (int64_t chunk = b; chunk < e; ++chunk) {
        int64_t count = 0;
        for (int64_t c = ccl(chunk); c < ccl(chunk + 1); ++c) {
          count += (cell_kx[c] != cell_kx[c - 1]);
        }
        col_chunk[chunk + 1] = count;
      }
    });
    for (int64_t chunk = 0; chunk < cell_chunks; ++chunk) {
      col_chunk[chunk + 1] += col_chunk[chunk];
    }
    const int32_t num_columns =
        static_cast<int32_t>(col_chunk[cell_chunks]) + 1;
    col_of_kx_start.assign(static_cast<size_t>(num_columns) + 1, 0);
    col_of_kx_start[num_columns] = num_cells;
    at::parallel_for(0, cell_chunks, 1, [&](int64_t b, int64_t e) {
      for (int64_t chunk = b; chunk < e; ++chunk) {
        int64_t col = col_chunk[chunk] + 1;
        for (int64_t c = ccl(chunk); c < ccl(chunk + 1); ++c) {
          if (cell_kx[c] != cell_kx[c - 1]) {
            col_of_kx_start[col] = static_cast<int32_t>(c);
            ++col;
          }
        }
      }
    });
  }
  const int32_t num_columns =
      static_cast<int32_t>(col_of_kx_start.size()) - 1;

  std::vector<int32_t> ngh(static_cast<size_t>(num_cells) * 25, -1);
  at::parallel_for(0, num_columns, 1, [&](int64_t b, int64_t e) {
    for (int64_t ci = b; ci < e; ++ci) {
      const int32_t a_lo = col_of_kx_start[ci], a_hi = col_of_kx_start[ci + 1];
      const int32_t kx = cell_kx[a_lo];
      for (int32_t dx = -2; dx <= 2; ++dx) {
        // Locate the kx + dx column: columns are sorted by uint32(kx), and
        // comparing in that order handles negative kx and the int-boundary
        // wraparound uniformly.
        auto col_less = [&](int32_t col, int32_t target_kx) {
          return static_cast<uint32_t>(cell_kx[col_of_kx_start[col]]) <
                 static_cast<uint32_t>(target_kx);
        };
        int32_t lo = 0, hi = num_columns;
        while (lo < hi) {
          int32_t mid = (lo + hi) >> 1;
          if (col_less(mid, kx + dx)) lo = mid + 1;
          else hi = mid;
        }
        if (lo >= num_columns || cell_kx[col_of_kx_start[lo]] != kx + dx) {
          continue;
        }

        const int32_t b_hi = col_of_kx_start[lo + 1];
        int32_t j = col_of_kx_start[lo];
        for (int32_t a = a_lo; a < a_hi; ++a) {
          const int32_t ky = cell_ky[a];
          while (j < b_hi && cell_ky[j] < ky - 2) ++j;
          int32_t* row = ngh.data() + size_t(a) * 25;
          for (int32_t jj = j; jj < b_hi && cell_ky[jj] <= ky + 2; ++jj) {
            row[(dx + 2) * 5 + (cell_ky[jj] - ky + 2)] = jj;
          }
        }
      }
    }
  });
  stage_end(3);

  // ---------------------------------------------------------------- stage 5
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
    // Center cell first, then ring 1, then ring 2 (see kCoreScanOrder): the
    // min_samples early exit usually fires before the ring-2 scans start.
    for (int32_t k = 0; k < 25; ++k) {
      const int32_t off = kCoreScanOrder[k];
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
  stage_end(4);

  // ---------------------------------------------------------------- stage 6
  // Per-cell point bounding boxes, [xmin, xmax, ymin, ymax] per cell. Used
  // below for conservative pruning: ring-2 neighbor offsets sit at a
  // geometric gap of up to exactly eps, so a dense cell pair there rarely
  // holds an eps-close pair, yet without a prune the scan runs the full
  // O(|c| * |nc|) pass. The box-box check rejects such pairs outright; the
  // per-point check caps a fruitless scan at O(|c|). Both skip only when no
  // point pair could be within eps, so the partition is unchanged.
  std::vector<float> cell_bbox(static_cast<size_t>(num_cells) * 4);
  at::parallel_for(0, num_cells, 256, [&](int64_t b, int64_t e) {
    for (int64_t c = b; c < e; ++c) {
      const int32_t s = cell_start[c], en = cell_start[c + 1];
      float xmin = xs[s], xmax = xs[s], ymin = ys[s], ymax = ys[s];
      for (int32_t i = s + 1; i < en; ++i) {
        xmin = std::min(xmin, xs[i]);
        xmax = std::max(xmax, xs[i]);
        ymin = std::min(ymin, ys[i]);
        ymax = std::max(ymax, ys[i]);
      }
      float* box = cell_bbox.data() + size_t(c) * 4;
      box[0] = xmin;
      box[1] = xmax;
      box[2] = ymin;
      box[3] = ymax;
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
    const float* bb_c = cell_bbox.data() + size_t(c) * 4;
    const int32_t s_c = cell_start[c], e_c = cell_start[c + 1];
    for (int32_t off = 0; off < 25; ++off) {
      if (off == 12) continue;
      const int32_t nc = row[off];
      if (nc <= static_cast<int32_t>(c)) continue;
      if (cell_status[nc] == 0) continue;
      if (uf.find(static_cast<int32_t>(c)) == uf.find(nc)) continue;
      const float* bb_n = cell_bbox.data() + size_t(nc) * 4;
      if (box_box_d2(bb_c, bb_n) > eps_sq) continue;

      const int32_t s_n = cell_start[nc], e_n = cell_start[nc + 1];
      bool found = false;
      for (int32_t i = s_c; i < e_c && !found; ++i) {
        if (!is_core[i]) continue;
        const float xi = xs[i], yi = ys[i];
        if (point_box_d2(xi, yi, bb_n) > eps_sq) continue;
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
  stage_end(5);

  // ---------------------------------------------------------------- stage 7
  // Renumber roots to dense cluster ids in cell order. Sequential.
  std::vector<int32_t> root_to_cluster(num_cells, kNoise);
  int32_t cid = 0;
  for (int32_t c = 0; c < num_cells; ++c) {
    if (cell_status[c] == 0) continue;
    const int32_t r = uf.find(c);
    if (root_to_cluster[r] == kNoise) root_to_cluster[r] = cid++;
  }
  stage_end(6);

  // ---------------------------------------------------------------- stage 8
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
  stage_end(7);

  // ---------------------------------------------------------------- stage 9
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
        if (point_box_d2(xi, yi, cell_bbox.data() + size_t(nc) * 4) > eps_sq)
          continue;
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
  stage_end(8);

  // ---------------------------------------------------------------- stage 10
  // Unpermute back to input order. Parallel.
  at::parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      out.labels[perm[i]] = sorted_labels[i];
    }
  });
  stage_end(9);
  out.n_clusters = cid;
  return out;
}

}  // namespace dbscan_grid
