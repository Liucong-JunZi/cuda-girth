#pragma once
#include <cstdint>
#include "cuda_girth/csr_graph.h"

namespace cuda_girth {

/// Compute exact girth for a single source vertex.
/// Returns 0 if the source belongs to no cycle (tree component).
int32_t girth_single_source(const CsrGraph& graph, int32_t source);

/// Compute exact girth from every vertex (batched multi-source lockstep BFS).
/// Returns 0 if the graph is a forest.
int32_t girth_all_sources(const CsrGraph& graph);

/// Compute exact girth for a batch of sources (lockstep, fills GPU).
int32_t girth_batched(const CsrGraph& graph, int32_t batch_start,
                       int32_t batch_size);

/// Compute per-source girth and write into out[0..n-1].
/// out[v] = local girth from source v (0 = no cycle found).
void girth_per_source(const CsrGraph& graph, int32_t* out);

}  // namespace cuda_girth
