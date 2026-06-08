#include "cuda_girth/csr_graph.h"
#include "cuda_girth/timer.h"
#include <algorithm>
#include <cstring>

namespace cuda_girth {

CsrGraph::~CsrGraph() {
    if (owns_memory) {
        CUDA_CHECK(cudaFree(row_ptr));
        CUDA_CHECK(cudaFree(col_ind));
    }
}

CsrGraph::CsrGraph(CsrGraph&& other) noexcept
    : num_vertices(other.num_vertices)
    , num_edges(other.num_edges)
    , row_ptr(other.row_ptr)
    , col_ind(other.col_ind)
    , owns_memory(other.owns_memory)
{
    other.owns_memory = false;
    other.row_ptr = nullptr;
    other.col_ind = nullptr;
}

CsrGraph& CsrGraph::operator=(CsrGraph&& other) noexcept {
    if (this != &other) {
        if (owns_memory) {
            cudaFree(row_ptr);
            cudaFree(col_ind);
        }
        num_vertices = other.num_vertices;
        num_edges    = other.num_edges;
        row_ptr      = other.row_ptr;
        col_ind      = other.col_ind;
        owns_memory  = other.owns_memory;
        other.owns_memory = false;
        other.row_ptr = nullptr;
        other.col_ind = nullptr;
    }
    return *this;
}

CsrGraph build_csr_from_edges(const std::vector<int32_t>& row,
                               const std::vector<int32_t>& col,
                               int32_t num_vertices,
                               bool is_symmetric)
{
    const size_t m = row.size();
    CsrGraph graph;
    graph.num_vertices = num_vertices;
    graph.owns_memory  = true;

    // Allocate unified memory
    CUDA_CHECK(cudaMallocManaged(&graph.row_ptr, (num_vertices + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(graph.row_ptr, 0, (num_vertices + 1) * sizeof(int32_t)));

    // Count degrees
    for (size_t i = 0; i < m; ++i) {
        graph.row_ptr[row[i] + 1]++;
        if (!is_symmetric && row[i] != col[i]) {
            graph.row_ptr[col[i] + 1]++;
        }
    }

    // Prefix sum → row_ptr
    for (int32_t v = 0; v < num_vertices; ++v) {
        graph.row_ptr[v + 1] += graph.row_ptr[v];
    }
    graph.num_edges = graph.row_ptr[num_vertices];

    CUDA_CHECK(cudaMallocManaged(&graph.col_ind, graph.num_edges * sizeof(int32_t)));

    // Second pass: fill col_ind
    std::vector<int32_t> offset(num_vertices, 0);
    for (size_t i = 0; i < m; ++i) {
        int32_t u = row[i], v = col[i];
        graph.col_ind[graph.row_ptr[u] + offset[u]++] = v;
        if (!is_symmetric && u != v) {
            graph.col_ind[graph.row_ptr[v] + offset[v]++] = u;
        }
    }

    // Prefetch to device
    int device = 0;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(graph.row_ptr, (num_vertices + 1) * sizeof(int32_t),
                          device, nullptr);
    cudaMemPrefetchAsync(graph.col_ind, graph.num_edges * sizeof(int32_t),
                          device, nullptr);

    return graph;
}

CsrGraph build_csr_undirected(int32_t num_vertices,
                               const std::vector<int32_t>& src,
                               const std::vector<int32_t>& dst)
{
    return build_csr_from_edges(src, dst, num_vertices, /*is_symmetric=*/false);
}

}  // namespace cuda_girth
