#pragma once
#include <cstdint>
#include "cuda_girth/csr_graph.h"

namespace cuda_girth {

/// Device-side BFS state for a single source.
/// Lives in unified memory; host reads early-stop flags.
struct BfsState {
    int32_t* levels;           // BFS level of each vertex (-1 = unvisited)
    int32_t* frontier;         // current level vertices, compact
    int32_t* next_frontier;    // next level vertices (produced by kernel)
    int32_t  frontier_size;
    int32_t  next_frontier_size;

    int32_t* next_frontier_counter;  // device-side atomic counter for kernel
    int32_t* global_min_cycle;       // atomicMin'd best cycle (2t+1 or 2t+2)

    int32_t  source;
    int32_t  num_vertices;
    int32_t  current_level;

    BfsState();
    ~BfsState();

    /// Allocate device-accessible storage for |V| vertices.
    void init(int32_t source_vertex, const CsrGraph& graph);

    /// Free all buffers.
    void free();

    // Move-only
    BfsState(BfsState&& other) noexcept;
    BfsState& operator=(BfsState&& other) noexcept;
    BfsState(const BfsState&) = delete;
    BfsState& operator=(const BfsState&) = delete;
};

}  // namespace cuda_girth
