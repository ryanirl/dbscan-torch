# CUDA non-determinism in dbscan-torch: investigation and fix

## TL;DR

The CUDA implementation of `dbscan2d` was non-deterministic: identical input
produced subtly different partitions across runs (~0.3-0.5% of points
landing in different clusters on dense, ambiguous inputs like uniform-random
points near the percolation threshold). DBSCAN as an algorithm is fully
deterministic for fixed `(X, eps, min_samples)`, so this was a real bug.

Root cause: a write-write race inside the `k_uf_flatten` kernel of the
cell-level union-find. The fix is **four lines in `csrc/cuda/dbscan_cuda.cu`**:
introduce a read-only variant of `uf_find` and use it in `k_uf_flatten`.

If you have an older copy of this code, you can apply the fix from the
[Patch](#patch) section at the bottom.

---

## 1. Symptom

Running `dbscan2d` on the same CUDA tensor twice produces label arrays that
encode different partitions:

```python
import torch
from dbscan_torch import dbscan2d

X = torch.randn(100_000, 2, device="cuda")
L0 = dbscan2d(X, eps=0.5, min_samples=5).cpu().numpy()
L1 = dbscan2d(X, eps=0.5, min_samples=5).cpu().numpy()
# Even after canonical relabeling (to normalize cluster-id permutations),
# L0 and L1 differ on ~0.3-0.5% of points on dense inputs.
```

On the synthetic `uniform` dataset (n=100k, 22k tiny clusters near percolation),
we measured ~450 points (0.45%) jumping between clusters across runs. Some of
those points moved between very differently-sized clusters (e.g., a 350-point
cluster in one run vs. a 9-point cluster in another), so this was not just
border-tiebreak jitter — the partition structure itself was changing.

The CPU implementation was fully deterministic on the same inputs.

## 2. Why it should be deterministic

DBSCAN's output is a function of `(X, eps, min_samples)`:

- The set of core points is deterministic (each point either has ≥ `min_samples`
  neighbors within `eps` or it doesn't).
- The connected components of "core points within eps" are deterministic.
- Border assignment is deterministic given a tiebreak rule.

The CUDA implementation's design explicitly aims for determinism: `k_border_assign`
uses a `(d2, perm[j])` lexicographic tiebreak, where `perm[j]` is the unique
original index of each candidate core. Even if the sort within a cell isn't
stable, the *set* of `(d2, perm)` tuples is invariant and `min` is
order-independent.

So in principle, every stage should be invariant under thread-scheduling
order. The empirical result said otherwise.

## 3. Investigation methodology

A "where does the divergence enter the pipeline?" approach. Built a debug
binding that materializes every per-stage intermediate, then diffed runs
stage by stage.

### 3.1 The debug binding

Added a `dbscan2d_cuda_debug` function alongside the production
`dbscan2d_cuda`. Both go through the same `dbscan2d_cuda_impl` static
helper, which takes an optional `DebugIntermediates*` argument that the
debug variant fills:

```cpp
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
```

In production the pointer is `nullptr` and the captures collapse to a
single predicted branch per stage (no allocation, no perf cost). In debug
mode, the captures are torch tensor handle copies (shared_ptr ref-bumps).

### 3.2 The stage-by-stage diff script

`evals/investigate_determinism.py` calls `dbscan2d_cuda_debug` twice on the
same input and checks bit-identity at every stage:

```
[ 0] labels                         DIFF   286/100000 differ (0.286%)
[ 1] keys_sorted                    OK
[ 2] perm                           OK
[ 3] xs                             OK
[ 4] ys                             OK
[ 5] is_core                        OK
[ 6] cell_start                     OK
[ 7] cluster_id_per_cell            DIFF   170/69442 differ (0.245%)
[ 8] uf                             DIFF   170/69442 differ (0.245%)
[ 9] sorted_labels_pre_border       DIFF
[10] sorted_labels_final            DIFF
```

This immediately localized the bug:

- The sort, gather, boundary scan, core-marking — all **bit-identical**
  across runs. So none of those were the source.
- The first kernel whose output differed was `uf` (the union-find roots).
  170 out of 69,442 cells ended up with different `uf[]` values across runs.
- The discrepancy propagates: `cluster_id_per_cell` reads `uf[c]` and indexes
  `root_id_scan[uf[c]]`, so a wrong `uf` entry yields a wrong cluster id.

### 3.3 Things we ruled out

Going in, the suspect list was:

1. **`thrust::sort_by_key` instability** — documented as not stable. For
   points sharing a cell key, the within-cell sort order could vary.
   **Ruled out**: `keys_sorted` and `perm` were bit-identical across runs.
   For our key space (`uint64`), thrust in practice uses a stable radix sort.
2. **`atomicCAS`-based unite in `k_unite_pairs`** — could the cell-graph
   edges vary across runs? **Ruled out**: the set of edges in the cell graph
   is a deterministic function of `is_core` and cell membership; even if the
   `atomicCAS` order varies, the final UF components are invariant (every
   redirect goes from higher to lower id, so the final root of any component
   is `min(component)`).
3. **Float-point ordering in distance comparisons** — could `d2 ≤ eps²` flip
   across runs? **Ruled out**: `d2` is a deterministic per-thread computation
   over the same physical point pair.

What was left: something inside the UF *flatten* stage was non-deterministic.

## 4. The bug

The pre-fix `k_uf_flatten` looked like this:

```cpp
__device__ __forceinline__ int32_t uf_find(int32_t* uf, int32_t x) {
  int32_t p = uf[x];
  while (p != x) {
    int32_t gp = uf[p];
    uf[x] = gp;          // <-- path-halving write
    x = p;
    p = gp;
  }
  return x;
}

__global__ void k_uf_flatten(int32_t* uf, int32_t num_cells) {
  int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_cells) return;
  uf[c] = uf_find(uf, c);  // <-- final assignment of the root
}
```

`uf_find` does path halving: as it walks from `x` toward the root, it writes
`uf[x] = grandparent(x)` along the way. After `uf_find` returns the root,
the caller writes `uf[c] = root`.

The race: **two different flatten threads can both write to `uf[c]`**.

- Thread `c` writes `uf[c] = root` as its final assignment.
- Any thread `d` where `c` is an ancestor of `d` *also* writes to `uf[c]`
  during its own `uf_find(d)` — specifically, when its path-halving walk
  passes through `c`, it writes `uf[c] = grandparent_of_c_at_that_moment`.

If thread `d`'s path-halving write to `uf[c]` lands **after** thread `c`'s
final assignment, `uf[c]` ends up pointing to whatever `grandparent_of_c`
was at the moment `d` read it — which is some valid ancestor of `c`, but
**not necessarily the root**.

### 4.1 Concrete example

Pre-flatten state (a chain `0 ← 1 ← 3 ← 5 ← 7 ← 9`, all in one component):

```
uf[0]=0, uf[1]=0, uf[3]=1, uf[5]=3, uf[7]=5, uf[9]=7
```

The root of the component is cell 0.

Thread 7 runs `uf_find(7)`:
- reads `uf[7]=5`, `uf[5]=3` → writes `uf[7]=3` (path halving)
- reads `uf[3]=1`, `uf[1]=0` → writes `uf[3]=0`
- … returns root 0
- assignment: `uf[7] = 0` ✓

Concurrently, thread 9 runs `uf_find(9)`. In its iteration 2 (x=5, p=3):
- reads `uf[5]=3` (the *original*, since thread 7 hasn't updated it yet)
- reads `uf[3]` → could be 1 (original) or 0 (if thread 7 has progressed)
- assume it reads 1 → `gp = 1`
- writes `uf[7] = ???` — wait, in iteration 2 of thread 9, `x=7` and we're
  about to write `uf[7] = gp = 1`.

So thread 9 writes `uf[7] = 1` somewhere in the middle of its walk. If this
write lands **after** thread 7's `uf[7] = 0` assignment, `uf[7]` ends up
pointing to cell 1, which is **not** a root.

### 4.2 Why the downstream code breaks

`k_cluster_id_per_cell` then does:

```cpp
cluster_id[c] = root_id_scan[uf[c]];
```

`root_id_scan` is the exclusive scan of `root_flag[c] = (uf[c] == c)`, so it
maps *roots* to dense cluster ids. If `uf[c]` is not a root,
`root_id_scan[uf[c]]` returns whatever value sits between consecutive roots
in the scan — which is a wrong cluster id for cell `c`.

Cell `c`'s points then inherit that wrong cluster id via `k_core_labels`, and
border points see the wrong "winning core" label via `k_border_assign`. The
result is a globally incorrect partition for any chain of cells deep enough
to trigger the race.

### 4.3 Why the race fires so rarely (~0.5% of cells)

The race requires:

1. Cell `c` is an interior node in the UF tree (not a leaf, not a root).
2. Some descendant `d` is also being flattened at the same wall time.
3. `d`'s path-halving write to `uf[c]` lands after `c`'s assignment.

Leaves never write to ancestors (they read but don't path-compress through
them — actually they do, but only after reading enough levels). Roots
trivially write themselves. The vulnerable middle layer is small. On
shallow trees, very few cells are vulnerable. On the `uniform` dataset
with ~22k clusters and ~3-4 cells/cluster on average, the affected fraction
matches what we see (~0.5%).

## 5. The fix

In `k_uf_flatten`, use a **read-only** find — one that walks to the root
without doing any path-halving writes along the way:

```cpp
__device__ __forceinline__ int32_t uf_find_readonly(const int32_t* uf,
                                                    int32_t x) {
  while (uf[x] != x) x = uf[x];
  return x;
}

__global__ void k_uf_flatten(int32_t* uf, int32_t num_cells) {
  int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_cells) return;
  uf[c] = uf_find_readonly(uf, c);  // <-- now: only writes its own slot
}
```

Each flatten thread now writes **only** `uf[c]` — its own slot. No two
threads ever write to the same `uf[]` index in this kernel. No write-write
race is possible.

The read-only walk still terminates at the correct root, even with stale
views: writes from other threads to `uf[]` (in either `k_unite_pairs` or
`k_uf_flatten`) can only redirect entries *toward* the root, never away
from it. So a walk that reads any valid prior state eventually reaches the
correct root.

### 5.1 Why the unite phase doesn't need this fix

The path-halving `uf_find` is still used inside `k_unite_pairs`'s `uf_unite`.
That's fine, because:

- The output of `uf_unite` is used only to feed `atomicCAS`, which is itself
  atomic and order-correct.
- The final state of `uf[]` after all unites is "every cell points to some
  ancestor in its component" — sufficient for `k_uf_flatten` to walk to the
  root.

The bug was specifically that `k_uf_flatten`, the kernel whose job is to
*finalize* `uf[c]` so downstream code can index `root_id_scan[uf[c]]`,
allowed concurrent path-halving writes to clobber its final assignment.

### 5.2 Performance

The read-only `uf_find_readonly` does *less* work than the path-halving
variant (no writes during the walk). In practice the flatten kernel is a
small fraction of total CUDA time, and the change is within measurement
noise on every config we tested (`blobs`, `dense_blobs`, `uniform`,
`worst_case` at N = 100k, 1M, 10M).

## 6. Verification

After the fix, the stage-by-stage diff is fully clean on every dataset:

```
$ python -m evals.investigate_determinism --dataset uniform --n 100000 --repeats 5
[ 0] labels                         OK
[ 1] keys_sorted                    OK
...
[10] sorted_labels_final            OK

All stages identical across runs. CUDA is deterministic on this input.
```

Same result on `dense_blobs`, `blobs`, `worst_case` at N ∈ {100k, 1M, 10M}.
The full pytest suite (33 tests) passes.

## 7. Patch

If you have an older copy of `csrc/cuda/dbscan_cuda.cu`, here is the minimal
diff to apply the fix. The change adds the read-only find and switches
`k_uf_flatten` to use it; nothing else moves.

```diff
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
+
+// Read-only walk to root. Used by k_uf_flatten only, where path-halving
+// writes would race: a flatten thread t_c writes uf[c] = root as its final
+// assignment, while a concurrent flatten thread t_d (with c on d's path)
+// can write uf[c] = some_ancestor as path halving inside its find. If t_d's
+// write lands after t_c's assignment, uf[c] ends up pointing to a non-root,
+// which silently corrupts cluster_id_per_cell. Read-only find sidesteps the
+// race: each flatten thread writes only its own uf[c] slot.
+__device__ __forceinline__ int32_t uf_find_readonly(const int32_t* uf,
+                                                    int32_t x) {
+  while (uf[x] != x) x = uf[x];
+  return x;
+}

 __device__ __forceinline__ void uf_unite(int32_t* uf, int32_t a, int32_t b) {
   ...
 }

-// Final compaction so uf[c] points directly to its root.
+// Final compaction so uf[c] points directly to its root. Uses
+// uf_find_readonly: see its comment for why path-halving find would race.
 __global__ void k_uf_flatten(int32_t* uf, int32_t num_cells) {
   int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
   if (c >= num_cells) return;
-  uf[c] = uf_find(uf, c);
+  uf[c] = uf_find_readonly(uf, c);
 }
```

That's the entire fix. After applying:

1. Rebuild: `pip install --no-build-isolation -e .`
2. Verify: run any DBSCAN call twice on the same CUDA input, confirm
   identical output.
3. Optional: run the stage-by-stage diff script
   (`python -m evals.investigate_determinism`) for stronger verification.

## 8. Lessons

- **Atomic operations don't make code race-free; they make individual
  operations atomic.** The bug here was between an atomic read-modify-write
  inside `uf_find` (the path-halving write) and a separate plain write (the
  final assignment in `k_uf_flatten`). Each individual write was fine; the
  race was at the level of *which thread's write wins*.

- **"Final-state-is-deterministic" is necessary but not sufficient.** The
  unite phase has the property that the final UF tree is uniquely determined
  (= every cell points to some ancestor of the component root). But the
  flatten phase needed the stronger property "uf[c] points directly to the
  root", which the path-halving variant did not guarantee under concurrent
  writes.

- **Per-stage intermediate diffing is a powerful debugging tool.** Once
  the debug binding was wired up, finding the responsible stage took one
  script run. Without it, the symptom (~0.5% partition drift) is too
  diffuse to debug from outputs alone.
