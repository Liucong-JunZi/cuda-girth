#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_girth/csr_graph.h"

namespace cuda_girth {

/// Read a simple edge-list file: first line "n m", then m lines of "u v".
/// Vertices are 0-based.
CsrGraph read_edge_list(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("Cannot open: " + path);

    int32_t n, m;
    if (fscanf(f, "%d %d", &n, &m) != 2) {
        fclose(f);
        throw std::runtime_error("Bad header in: " + path);
    }

    std::vector<int32_t> src(m), dst(m);
    for (int32_t i = 0; i < m; ++i) {
        if (fscanf(f, "%d %d", &src[i], &dst[i]) != 2) {
            fclose(f);
            throw std::runtime_error("Truncated file: " + path +
                                     " at edge " + std::to_string(i));
        }
    }
    fclose(f);

    return build_csr_undirected(n, src, dst);
}

/// Write CSR in plain text for debugging.
void write_csr_text(const CsrGraph& graph, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("Cannot write: " + path);

    fprintf(f, "%d %d\n", graph.num_vertices, graph.num_edges);
    for (int32_t v = 0; v < graph.num_vertices; ++v) {
        for (int32_t e = graph.begin(v); e < graph.end(v); ++e) {
            fprintf(f, "%d %d\n", v, graph.col_ind[e]);
        }
    }
    fclose(f);
}

}  // namespace cuda_girth
