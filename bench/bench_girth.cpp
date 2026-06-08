#include <cstdio>
#include <cstdlib>
#include <string>
#include "cuda_girth/girth_engine.h"
#include "cuda_girth/csr_graph.h"
#include "cuda_girth/timer.h"

namespace cg = cuda_girth;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph.txt> [num_warmup] [num_runs]\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];
    int warmup = (argc >= 3) ? atoi(argv[2]) : 2;
    int runs   = (argc >= 4) ? atoi(argv[3]) : 5;

    cg::CsrGraph graph = cg::read_edge_list(path);
    fprintf(stderr, "Graph: |V|=%d  |E|=%d  warmup=%d  runs=%d\n",
            graph.num_vertices, graph.num_edges / 2, warmup, runs);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        cg::girth_all_sources(graph);
        cudaDeviceSynchronize();
    }

    // Timed runs
    double total_ms = 0.0;
    for (int i = 0; i < runs; ++i) {
        cg::GpuTimer timer;
        timer.start();
        int32_t g = cg::girth_all_sources(graph);
        timer.stop();
        float ms = timer.sync();

        total_ms += ms;
        fprintf(stdout, "  run %d: girth=%d  %.3f ms\n", i + 1, g, ms);
    }

    fprintf(stdout, "avg: %.3f ms\n", total_ms / runs);
    return 0;
}
