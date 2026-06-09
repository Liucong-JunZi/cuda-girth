#include <cstdio>
#include <cstdlib>
#include <string>
#include "cuda_girth/girth_engine.h"
#include "cuda_girth/csr_graph.h"
#include "cuda_girth/graph_io.h"
#include "cuda_girth/timer.h"

namespace cg = cuda_girth;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph.txt> [source_vertex]\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];
    int source = (argc >= 3) ? atoi(argv[2]) : -1;

    fprintf(stderr, "Loading graph: %s\n", path.c_str());
    cg::CsrGraph graph = cg::read_edge_list(path);
    fprintf(stderr, "  vertices = %d, edges = %d\n",
            graph.num_vertices, graph.num_edges);

    {
        cg::CpuTimer timer("girth");
        if (source >= 0) {
            int32_t g = cg::girth_single_source(graph, source);
            fprintf(stdout, "girth(%d) = %d\n", source, g);
        } else {
            int32_t g = cg::girth_all_sources(graph);
            fprintf(stdout, "girth = %d\n", g);
        }
    }

    return 0;
}
