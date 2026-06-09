/// bfs_level.cu
///
/// Two kernel variants:
///   1. bfs_level_kernel        — single-source BFS (for girth_single_source)
///   2. bfs_level_batched_kernel — multi-source lockstep BFS (for girth_all_sources)
///
/// Both detect intra-level (2t+1) and cross-level (2t+2) non-tree edges,
/// matching the lemma's strategy.

#include "cuda_girth/csr_graph.h"
#include "cuda_girth/bfs_state.h"
#include "cuda_girth/batch_state.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace cuda_girth {

static constexpr int WARP_SIZE  = 32;
static constexpr int INF_CYCLE  = INT32_MAX;

// ===========================================================================
// Single-source kernel (unchanged logic)
// ===========================================================================
__global__ void bfs_level_kernel(
    const CsrGraph graph,
    const int32_t* __restrict__ frontier,
    int32_t         frontier_size,
    int32_t         current_level,
    int32_t* __restrict__ levels,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t* __restrict__ global_min_cycle
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = tid / WARP_SIZE;
    int lid = tid % WARP_SIZE;

    if (wid >= frontier_size) return;

    int src = frontier[wid];
    int e_start = graph.begin(src);
    int e_end   = graph.end(src);

    int local_best = INF_CYCLE;

    for (int e = e_start + lid; e < e_end; e += WARP_SIZE) {
        int nbr = graph.col_ind[e];
        if (nbr == src) continue;

        int lev = levels[nbr];

        if (lev == -1) {
            int old = atomicCAS(&levels[nbr], -1, current_level + 1);
            if (old == -1) {
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = nbr;
            } else if (old == current_level) {
                local_best = min(local_best, 2 * current_level + 1);
            } else if (old == current_level + 1) {
                local_best = min(local_best, 2 * current_level + 2);
            }
        }
        else if (lev == current_level) {
            local_best = min(local_best, 2 * current_level + 1);
        }
        else if (lev == current_level + 1) {
            local_best = min(local_best, 2 * current_level + 2);
        }
    }

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    int warp_best = warp.reduce(local_best, cg::less<int>());

    if (lid == 0 && warp_best < INF_CYCLE) {
        atomicMin(global_min_cycle, warp_best);
    }
}

// ===========================================================================
// Multi-source batched kernel
//
// Each warp handles one (source_id, vertex) pair.
// levels / next_frontier / next_size / min_cycle are flat arrays
// indexed by [source_id * num_vertices + vertex] or [source_id].
// ===========================================================================
__global__ void bfs_level_batched_kernel(
    const CsrGraph graph,
    const int32_t* __restrict__ source_ids,      // [total_frontier]
    const int32_t* __restrict__ frontier,         // [total_frontier]
    int32_t         total_frontier_size,
    int32_t         current_level,
    int32_t         num_vertices,
    int32_t* __restrict__ levels,                 // [num_sources * num_vertices]
    int32_t* __restrict__ next_frontier,          // [num_sources * num_vertices]
    int32_t* __restrict__ next_size,              // [num_sources]
    int32_t* __restrict__ min_cycle               // [num_sources]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = tid / WARP_SIZE;
    int lid = tid % WARP_SIZE;

    if (wid >= total_frontier_size) return;

    int src_id = source_ids[wid];                // which source
    int vertex = frontier[wid];                  // which vertex

    int* my_levels  = levels  + src_id * num_vertices;
    int* my_next    = next_frontier + src_id * num_vertices;
    int* my_nsize   = next_size + src_id;
    int* my_min     = min_cycle + src_id;

    int e_start = graph.begin(vertex);
    int e_end   = graph.end(vertex);

    int local_best = INF_CYCLE;

    for (int e = e_start + lid; e < e_end; e += WARP_SIZE) {
        int nbr = graph.col_ind[e];
        if (nbr == vertex) continue;

        int lev = my_levels[nbr];

        if (lev == -1) {
            int old = atomicCAS(&my_levels[nbr], -1, current_level + 1);
            if (old == -1) {
                int pos = atomicAdd(my_nsize, 1);
                my_next[pos] = nbr;
            } else if (old == current_level) {
                local_best = min(local_best, 2 * current_level + 1);
            } else if (old == current_level + 1) {
                local_best = min(local_best, 2 * current_level + 2);
            }
        }
        else if (lev == current_level) {
            local_best = min(local_best, 2 * current_level + 1);
        }
        else if (lev == current_level + 1) {
            local_best = min(local_best, 2 * current_level + 2);
        }
    }

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    int warp_best = warp.reduce(local_best, cg::less<int>());

    if (lid == 0 && warp_best < INF_CYCLE) {
        atomicMin(my_min, warp_best);
    }
}

// ===========================================================================
// Launchers
// ===========================================================================

void launch_bfs_level(const CsrGraph& graph, BfsState& state, cudaStream_t stream) {
    if (state.frontier_size == 0) return;

    static int block_size = 0;
    if (block_size == 0) {
        int min_grid;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid, &block_size,
            bfs_level_kernel, 0);
    }

    int num_warps  = state.frontier_size;
    int num_blocks = (num_warps * WARP_SIZE + block_size - 1) / block_size;

    cudaMemsetAsync(state.next_frontier_counter, 0, sizeof(int32_t), stream);

    bfs_level_kernel<<<num_blocks, block_size, 0, stream>>>(
        graph,
        state.frontier,
        state.frontier_size,
        state.current_level,
        state.levels,
        state.next_frontier,
        state.next_frontier_counter,
        state.global_min_cycle
    );
}

void launch_bfs_level_batched(const CsrGraph& graph, BatchState& batch,
                               cudaStream_t stream) {
    if (batch.total_frontier_size == 0) return;

    static int block_size = 0;
    if (block_size == 0) {
        int min_grid;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid, &block_size,
            bfs_level_batched_kernel, 0);
    }

    int num_warps  = batch.total_frontier_size;
    int num_blocks = (num_warps * WARP_SIZE + block_size - 1) / block_size;

    // Reset per-source next-size counters
    cudaMemsetAsync(batch.next_size, 0, batch.num_sources * sizeof(int32_t), stream);

    bfs_level_batched_kernel<<<num_blocks, block_size, 0, stream>>>(
        graph,
        batch.source_ids,
        batch.frontier,
        batch.total_frontier_size,
        batch.current_level,
        batch.num_vertices,
        batch.levels,
        batch.next_frontier,
        batch.next_size,
        batch.min_cycle
    );
}

}  // namespace cuda_girth
