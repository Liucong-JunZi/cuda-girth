# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
# C++ build (debug, with tests)
cmake -B build -S . -DCUDA_GIRTH_BUILD_PYTHON=OFF
cmake --build build -j

# Run tests
cd build && ctest --output-on-failure

# Run single test
./build/test_girth
./build/test_csr

# CLI: compute girth of a graph file (edge-list: first line "n m", then m lines "u v")
./build/girth_cli test/graphs/petersen.txt

# Python build & install
pip install -e .

# Python usage
python -c "import networkx as nx; from cuda_girth import girth; print(girth(nx.petersen_graph()))"
```

## Architecture

### Theoretical foundation

The entire algorithm rests on the **Cross-Level Non-Tree Edge Local Optimality Lemma** (`lemma_cross_level_optimality.md`). During BFS from source `s`, at level `t`:

| Non-tree edge type | Condition | Cycle length | Action |
|---|---|---|---|
| Intra-level | ℓ(u)=ℓ(v)=t | 2t+1 | Return immediately |
| Cross-level | ℓ(u)=t, ℓ(v)=t+1 | 2t+2 | Finish scanning level t, then return |
| Deeper levels | ≥t+1 | ≥2t+3 | Never search — can't beat 2t+2 |

### Two code paths

1. **Single-source** (`girth_single_source`): `BfsState` + `bfs_level_kernel`. One warp per frontier vertex, host loop drives level-by-level advancement. Used by `girth_per_source`.

2. **Multi-source batched** (`girth_all_sources`): `BatchState` + `bfs_level_batched_kernel`. Multiple sources run lockstep BFS — each warp handles a `(source_id, vertex)` pair. All sources' levels are packed into flattened arrays (`levels[S*V]`). Per-source lemma pruning: once `min_cycle[s] < INF`, source `s` exits. Batch size is auto-tuned from available GPU memory. Used by `girth_all_sources`.

### Data flow

```
NetworkX Graph → scipy CSR → NumPy arrays (indptr, indices, int32)
                                       │
                              pybind11 _bindings.cpp
                              CSR → COO → build_csr_from_edges(is_symmetric=true)
                                       │
                              CsrGraph (unified memory)
                              row_ptr, col_ind, num_vertices
                                       │
                    ┌──────────────────┴───────────────────┐
                    │                                      │
           girth_single_source                    girth_all_sources
           (BfsState)                             (BatchState)
           per-source BFS                         multi-source lockstep BFS
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       │
                                  Python int (girth)
```

### Key files

| File | Role |
|---|---|
| `src/kernel/bfs_level.cu` | Both kernel variants: single-source and batched. Detects intra-level (2t+1) and cross-level (2t+2) non-tree edges via warp-efficient CSR traversal + atomicCAS with dual-path detection (snapshot read + CAS return value). |
| `src/host/girth_engine.cpp` | Host loop for single-source: `while(frontier>0) { launch → check min_cycle → advance or break }`. |
| `src/host/batch_engine.cpp` | Host loop for batched multi-source: `while(total_frontier>0) { launch → per-source check → pack next frontier }`. Contains `calc_batch_size()` for auto-tuning. |
| `src/host/csr_builder.cpp` | Edge list → CSR in unified memory. `build_csr_from_edges(is_symmetric)` controls whether reverse edges are auto-generated. |
| `python/cuda_girth/bridge.py` | `nx_to_csr(G)` — NetworkX → scipy CSR → NumPy int32 arrays. |
| `python/cuda_girth/_bindings.cpp` | pybind11: CSR arrays → COO → `build_csr_from_edges(is_symmetric=true)` → call girth engine. `is_symmetric=true` because NetworkX CSR is already symmetric. |

### Memory conventions

- All GPU-accessible data uses **unified memory** (`cudaMallocManaged`). Host reads kernel outputs directly — no explicit `cudaMemcpy`.
- `CsrGraph` owns its memory, move-only (no copy).
- Per-source arrays in `BatchState` are flat: `levels[source_id * V + vertex]`.
- Compact frontier for batched kernel: `source_ids[i]` + `frontier[i]`, length `total_frontier_size`, rebuilt each level by host packing.

### Idiosyncrasies

- `clang` diagnostics on `.cu` and CUDA headers are expected noise — the project compiles with `nvcc` via CMake, which sets up CUDA include paths correctly.
- `__host__ __device__` qualifiers break non-CUDA linters. `include/cuda_girth/portability.h` defines empty macros for `#ifndef __CUDACC__`.
- The kernel's non-tree edge detection uses a **dual-path** strategy: snapshot read of `levels[nbr]` (fast, may be stale across blocks) + `atomicCAS` return value (strongly consistent, catches races). This ensures no missed detection without global barriers.
- ISPC/vectorization notes: warp = 32 threads (hardware), block size auto-tuned via `cudaOccupancyMaxPotentialBlockSize`.

---

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
