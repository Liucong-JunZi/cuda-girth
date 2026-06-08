#include "cuda_girth/bfs_state.h"
#include "cuda_girth/timer.h"
#include <cstring>

namespace cuda_girth {

BfsState::BfsState()
    : levels(nullptr)
    , frontier(nullptr)
    , next_frontier(nullptr)
    , frontier_size(0)
    , next_frontier_size(0)
    , next_frontier_counter(nullptr)
    , global_min_cycle(nullptr)
    , source(0)
    , num_vertices(0)
    , current_level(0)
{}

BfsState::~BfsState() {
    free();
}

void BfsState::init(int32_t source_vertex, const CsrGraph& graph) {
    free();
    source        = source_vertex;
    num_vertices  = graph.num_vertices;
    current_level = 0;

    int32_t n = num_vertices;

    // Allocate all buffers in unified memory
    CUDA_CHECK(cudaMallocManaged(&levels, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&frontier, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&next_frontier, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&next_frontier_counter, sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&global_min_cycle, sizeof(int32_t)));

    // Initialize
    for (int32_t i = 0; i < n; ++i) levels[i] = -1;
    levels[source]            = 0;
    frontier[0]               = source;
    frontier_size             = 1;
    next_frontier_size        = 0;
    *next_frontier_counter    = 0;
    *global_min_cycle         = INT32_MAX;

    // Prefetch to device
    int device = 0;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(levels, n * sizeof(int32_t), device, nullptr);
}

void BfsState::free() {
    auto safe_free = [](int32_t*& ptr) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    };
    safe_free(levels);
    safe_free(frontier);
    safe_free(next_frontier);
    safe_free(next_frontier_counter);
    safe_free(global_min_cycle);
}

BfsState::BfsState(BfsState&& other) noexcept
    : levels(other.levels)
    , frontier(other.frontier)
    , next_frontier(other.next_frontier)
    , frontier_size(other.frontier_size)
    , next_frontier_size(other.next_frontier_size)
    , next_frontier_counter(other.next_frontier_counter)
    , global_min_cycle(other.global_min_cycle)
    , source(other.source)
    , num_vertices(other.num_vertices)
    , current_level(other.current_level)
{
    other.levels                = nullptr;
    other.frontier              = nullptr;
    other.next_frontier         = nullptr;
    other.next_frontier_counter = nullptr;
    other.global_min_cycle      = nullptr;
}

BfsState& BfsState::operator=(BfsState&& other) noexcept {
    if (this != &other) {
        free();
        levels                = other.levels;
        frontier              = other.frontier;
        next_frontier         = other.next_frontier;
        frontier_size         = other.frontier_size;
        next_frontier_size    = other.next_frontier_size;
        next_frontier_counter = other.next_frontier_counter;
        global_min_cycle      = other.global_min_cycle;
        source                = other.source;
        num_vertices          = other.num_vertices;
        current_level         = other.current_level;

        other.levels                = nullptr;
        other.frontier              = nullptr;
        other.next_frontier         = nullptr;
        other.next_frontier_counter = nullptr;
        other.global_min_cycle      = nullptr;
    }
    return *this;
}

}  // namespace cuda_girth
