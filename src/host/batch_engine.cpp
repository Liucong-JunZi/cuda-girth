/// batch_engine.cpp
///
/// Host-side multi-source lockstep BFS engine.
///
/// Strategy:
///   1. Pack S source vertices into a single batch.
///   2. All sources advance level-by-level together — one kernel launch per level.
///   3. Each warp handles one (source_id, vertex) pair.
///   4. Per-source lemma pruning: once min_cycle[s] < INF, source s is done.
///   5. After kernel: read per-source results, pack next frontier for active sources.

#include "cuda_girth/girth_engine.h"
#include "cuda_girth/batch_state.h"
#include "cuda_girth/timer.h"
#include <algorithm>
#include <cstring>

namespace cuda_girth {

// ---- kernel launchers ----
void launch_bfs_level_batched(const CsrGraph& graph, BatchState& batch,
                               cudaStream_t stream);

// ===========================================================================
// Memory helpers
// ===========================================================================
static size_t gpu_free_mem() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

static int32_t calc_batch_size(int32_t num_vertices) {
    // Each source costs ~3 * V * sizeof(int32) bytes (levels + frontier buffers).
    // Leave 20% margin.
    size_t free_bytes = gpu_free_mem();
    size_t per_source = 3ull * num_vertices * sizeof(int32_t);
    if (per_source == 0) per_source = 1;
    int32_t max_by_mem = static_cast<int32_t>((free_bytes * 8 / 10) / per_source);
    if (max_by_mem < 1) max_by_mem = 1;
    return std::min(max_by_mem, num_vertices);
}

// ===========================================================================
// BatchState implementation
// ===========================================================================
BatchState::BatchState()
    : num_sources(0), num_vertices(0), current_level(0),
      levels(nullptr), min_cycle(nullptr), active(nullptr),
      frontier_size(nullptr), next_size(nullptr), next_frontier(nullptr),
      source_ids(nullptr), frontier(nullptr), total_frontier_size(0),
      batch_start(0)
{}

BatchState::~BatchState() { free(); }

void BatchState::init(int32_t batch_start_, int32_t batch_size,
                       const CsrGraph& graph) {
    free();
    num_sources   = batch_size;
    num_vertices  = graph.num_vertices;
    current_level = 0;
    batch_start   = batch_start_;

    int32_t S = num_sources;
    int32_t V = num_vertices;

    CUDA_CHECK(cudaMallocManaged(&levels,       S * V * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&min_cycle,    S * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&active,       S * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&frontier_size, S * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&next_size,    S * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&next_frontier, S * V * sizeof(int32_t)));

    // source_ids and frontier — max possible size is S * V (all vertices in frontier)
    CUDA_CHECK(cudaMallocManaged(&source_ids, S * V * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&frontier,    S * V * sizeof(int32_t)));

    // Initialize per-source state
    // 1. Bulk memset levels to -1 on device (avoids S*V host writes)
    CUDA_CHECK(cudaMemset(levels, 0xFF, S * V * sizeof(int32_t)));

    // 2. Per-source setup: source vertex → level 0, initial frontier
    for (int32_t s = 0; s < S; ++s) {
        int32_t global_src = batch_start + s;
        levels[s * V + global_src] = 0;

        min_cycle[s]     = INT32_MAX;
        active[s]        = 1;
        frontier_size[s] = 1;
        next_size[s]     = 0;

        source_ids[s] = s;
        frontier[s]   = global_src;
    }
    total_frontier_size = S;

    // 3. Prefetch to device
    int device = 0;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(levels,       S * V * sizeof(int32_t), device, nullptr);
    cudaMemPrefetchAsync(min_cycle,    S * sizeof(int32_t),     device, nullptr);
    cudaMemPrefetchAsync(next_size,    S * sizeof(int32_t),     device, nullptr);
    cudaMemPrefetchAsync(source_ids,   S * V * sizeof(int32_t), device, nullptr);
    cudaMemPrefetchAsync(frontier,     S * V * sizeof(int32_t), device, nullptr);
}

void BatchState::free() {
    auto sf = [](int32_t*& p) { if (p) { cudaFree(p); p = nullptr; } };
    sf(levels);       sf(min_cycle);   sf(active);
    sf(frontier_size); sf(next_size);  sf(next_frontier);
    sf(source_ids);   sf(frontier);
}

BatchState::BatchState(BatchState&& o) noexcept
    : num_sources(o.num_sources), num_vertices(o.num_vertices),
      current_level(o.current_level),
      levels(o.levels), min_cycle(o.min_cycle), active(o.active),
      frontier_size(o.frontier_size), next_size(o.next_size),
      next_frontier(o.next_frontier),
      source_ids(o.source_ids), frontier(o.frontier),
      total_frontier_size(o.total_frontier_size), batch_start(o.batch_start)
{
    o.levels = o.min_cycle = o.active = nullptr;
    o.frontier_size = o.next_size = o.next_frontier = nullptr;
    o.source_ids = o.frontier = nullptr;
}

BatchState& BatchState::operator=(BatchState&& o) noexcept {
    if (this != &o) {
        free();
        num_sources = o.num_sources; num_vertices = o.num_vertices;
        current_level = o.current_level;
        levels = o.levels; min_cycle = o.min_cycle; active = o.active;
        frontier_size = o.frontier_size; next_size = o.next_size;
        next_frontier = o.next_frontier;
        source_ids = o.source_ids; frontier = o.frontier;
        total_frontier_size = o.total_frontier_size; batch_start = o.batch_start;
        o.levels = o.min_cycle = o.active = nullptr;
        o.frontier_size = o.next_size = o.next_frontier = nullptr;
        o.source_ids = o.frontier = nullptr;
    }
    return *this;
}

// ===========================================================================
// Batched girth computation
// ===========================================================================
int32_t girth_batched(const CsrGraph& graph, int32_t batch_start,
                       int32_t batch_size) {
    BatchState batch;
    batch.init(batch_start, batch_size, graph);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int32_t g_min = INT32_MAX;

    while (batch.total_frontier_size > 0) {
        // ---- one kernel launch for ALL sources at this level ----
        launch_bfs_level_batched(graph, batch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ---- read per-source results ----
        for (int32_t s = 0; s < batch_size; ++s) {
            if (!batch.active[s]) continue;

            if (batch.min_cycle[s] < INT32_MAX) {
                // Lemma: found a cycle for this source — deeper levels can't beat it
                batch.active[s] = 0;
                if (batch.min_cycle[s] < g_min) {
                    g_min = batch.min_cycle[s];
                }
            }
        }

        // ---- build next level's compact frontier ----
        batch.total_frontier_size = 0;
        for (int32_t s = 0; s < batch_size; ++s) {
            if (!batch.active[s]) continue;

            int32_t ns = batch.next_size[s];
            if (ns == 0) {
                // No next frontier — source hit a dead end (tree component)
                batch.active[s] = 0;
                continue;
            }

            // Copy this source's next_frontier into the compact global frontier
            int32_t* src_next = batch.next_frontier + s * batch.num_vertices;
            for (int32_t i = 0; i < ns; ++i) {
                int32_t pos = batch.total_frontier_size++;
                batch.source_ids[pos] = s;
                batch.frontier[pos]    = src_next[i];
            }

            batch.frontier_size[s] = ns;
            batch.next_size[s]     = 0;   // reset for next kernel
        }

        batch.current_level++;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return (g_min == INT32_MAX) ? 0 : g_min;
}

// ===========================================================================
// All sources (batched)
// ===========================================================================
int32_t girth_all_sources_batched(const CsrGraph& graph) {
    int32_t V = graph.num_vertices;
    int32_t batch_sz = calc_batch_size(V);

    int32_t g_min = INT32_MAX;
    for (int32_t start = 0; start < V; start += batch_sz) {
        int32_t n = std::min(batch_sz, V - start);
        int32_t g = girth_batched(graph, start, n);
        if (g > 0 && g < g_min) {
            g_min = g;
            if (g_min == 3) break;  // triangle → can't be shorter
        }
    }
    return (g_min == INT32_MAX) ? 0 : g_min;
}

}  // namespace cuda_girth
