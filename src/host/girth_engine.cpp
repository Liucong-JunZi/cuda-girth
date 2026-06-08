/// girth_engine.cpp
///
/// Host-side girth computation loop.
///
/// Strategy (per source s):
///   1. BFS level-by-level.
///   2. At level t, launch one kernel that processes ALL frontier vertices.
///      The kernel simultaneously:
///        - expands the BFS tree (tree edges)
///        - detects intra-level  non-tree edges → cycle 2t+1
///        - detects cross-level non-tree edges → cycle 2t+2
///      Both are written to global_min_cycle via atomicMin.
///   3. After kernel completes:
///        - If global_min_cycle < INF: level t is fully scanned, return it.
///          (2t+1 beats 2t+2, atomicMin already picked the shorter one.)
///        - Otherwise: advance to t+1.
///   4. Repeat until frontier is empty (tree → girth = 0).
///
///   This directly implements the lemma's "finish-scanning-the-current-level"
///   policy — the parallel kernel naturally scans the whole level.

#include "cuda_girth/girth_engine.h"
#include "cuda_girth/bfs_state.h"
#include "cuda_girth/timer.h"
#include <algorithm>
#include <cstring>

namespace cuda_girth {

// ---- kernel launchers (defined in bfs_level.cu) ----
void launch_bfs_level(const CsrGraph& graph, BfsState& state, cudaStream_t stream);

// ===========================================================================
// single source
// ===========================================================================
int32_t girth_single_source(const CsrGraph& graph, int32_t source) {
    BfsState state;
    state.init(source, graph);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int32_t result = 0;

    while (state.frontier_size > 0) {
        // ---- expand level t → discover non-tree edges ----
        launch_bfs_level(graph, state, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ---- read results from unified memory ----
        int best = *state.global_min_cycle;

        if (best < INT32_MAX) {
            // Kernel scanned ALL of level t.
            // best = min(2t+1, 2t+2) — atomicMin picks the shorter.
            // Lemma: no deeper level can beat this.
            result = best;
            break;
        }

        // ---- no cycle yet: read next-frontier size, advance to t+1 ----
        state.next_frontier_size = *state.next_frontier_counter;
        state.current_level++;
        state.frontier_size = state.next_frontier_size;
        state.next_frontier_size = 0;
        std::swap(state.frontier, state.next_frontier);

        // Reset detectors for next level
        *state.global_min_cycle = INT32_MAX;
        *state.next_frontier_counter = 0;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return result;
}

// ===========================================================================
// all sources (batched multi-source lockstep BFS)
// ===========================================================================

// Declared in batch_engine.cpp
int32_t girth_all_sources_batched(const CsrGraph& graph);

int32_t girth_all_sources(const CsrGraph& graph) {
    return girth_all_sources_batched(graph);
}

// ===========================================================================
// per source
// ===========================================================================
void girth_per_source(const CsrGraph& graph, int32_t* out) {
    // TODO: use batched version for efficiency
    for (int32_t s = 0; s < graph.num_vertices; ++s) {
        out[s] = girth_single_source(graph, s);
    }
}

}  // namespace cuda_girth
