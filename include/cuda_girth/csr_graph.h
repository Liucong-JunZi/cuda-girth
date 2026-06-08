#pragma once
#include <cstdint>
#include <vector>

namespace cuda_girth {

/// Compressed Sparse Row representation, zero-copy portable to device.
/// Allocates unified memory so both host and kernel can access.
struct CsrGraph {
    int32_t  num_vertices;
    int32_t  num_edges;       // directed count (== 2 * |E| for undirected)
    int32_t* row_ptr;         // size num_vertices + 1
    int32_t* col_ind;         // size num_edges
    bool     owns_memory;

    CsrGraph() : num_vertices(0), num_edges(0),
                 row_ptr(nullptr), col_ind(nullptr), owns_memory(false) {}

    ~CsrGraph();

    // Move-only
    CsrGraph(CsrGraph&& other) noexcept;
    CsrGraph& operator=(CsrGraph&& other) noexcept;
    CsrGraph(const CsrGraph&) = delete;
    CsrGraph& operator=(const CsrGraph&) = delete;

    /// Number of neighbours of vertex v.
    __host__ __device__ inline int32_t degree(int32_t v) const {
        return row_ptr[v + 1] - row_ptr[v];
    }

    /// First edge index of vertex v.
    __host__ __device__ inline int32_t begin(int32_t v) const {
        return row_ptr[v];
    }

    /// Past-the-end edge index of vertex v.
    __host__ __device__ inline int32_t end(int32_t v) const {
        return row_ptr[v + 1];
    }
};

/// Build a CSR from an edge list.
/// Undirected: each (u, v) auto-generates (v, u) unless already symmetric.
CsrGraph build_csr_from_edges(const std::vector<int32_t>& row,
                               const std::vector<int32_t>& col,
                               int32_t num_vertices,
                               bool is_symmetric = false);

/// Build CSR directly from symmetric COO (one direction per edge).
CsrGraph build_csr_undirected(int32_t num_vertices,
                               const std::vector<int32_t>& src,
                               const std::vector<int32_t>& dst);

}  // namespace cuda_girth
