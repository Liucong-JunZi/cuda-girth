#pragma once
#include <cstdint>
#include "cuda_girth/csr_graph.h"

namespace cuda_girth {

/// Multi-source BFS state for lockstep batched execution.
/// S sources share a single kernel launch per level.
struct BatchState {
    int32_t num_sources;
    int32_t num_vertices;
    int32_t current_level;

    // ---- per-source arrays (flat, unified memory) ----
    int32_t* levels;              // [S * V]       BFS level per (source, vertex)
    int32_t* min_cycle;           // [S]           result per source
    int32_t* active;              // [S]           1=running, 0=done
    int32_t* frontier_size;       // [S]           current frontier size
    int32_t* next_size;           // [S]           next frontier size (device counter)
    int32_t* next_frontier;       // [S * V]       next level output buffer

    // ---- compact frontier (rebuilt each level by host) ----
    int32_t* source_ids;          // [total_frontier]   per-warp source owner
    int32_t* frontier;            // [total_frontier]   per-warp vertex
    int32_t  total_frontier_size;

    // ---- source vertex range ----
    int32_t batch_start;          // first source id in this batch

    BatchState();
    ~BatchState();

    void init(int32_t batch_start_, int32_t batch_size,
              const CsrGraph& graph);
    void free();

    BatchState(BatchState&&) noexcept;
    BatchState& operator=(BatchState&&) noexcept;
    BatchState(const BatchState&) = delete;
    BatchState& operator=(const BatchState&) = delete;
};

}  // namespace cuda_girth
