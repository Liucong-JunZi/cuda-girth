#include <gtest/gtest.h>
#include "cuda_girth/csr_graph.h"

using namespace cuda_girth;

TEST(CsrBuilder, Empty) {
    CsrGraph g = build_csr_undirected(0, {}, {});
    EXPECT_EQ(g.num_vertices, 0);
    EXPECT_EQ(g.num_edges, 0);
}

TEST(CsrBuilder, Triangle) {
    // Triangle: 0-1, 1-2, 2-0
    std::vector<int32_t> src = {0, 1, 2};
    std::vector<int32_t> dst = {1, 2, 0};

    CsrGraph g = build_csr_undirected(3, src, dst);
    EXPECT_EQ(g.num_vertices, 3);
    EXPECT_EQ(g.num_edges, 6);  // undirected: 2 * 3

    // Each vertex should have degree 2
    for (int v = 0; v < 3; ++v) {
        EXPECT_EQ(g.degree(v), 2);
    }
}

TEST(CsrBuilder, Square) {
    // Square: 0-1-2-3-0
    std::vector<int32_t> src = {0, 1, 2, 3};
    std::vector<int32_t> dst = {1, 2, 3, 0};

    CsrGraph g = build_csr_undirected(4, src, dst);
    EXPECT_EQ(g.num_vertices, 4);
    EXPECT_EQ(g.num_edges, 8);

    for (int v = 0; v < 4; ++v) {
        EXPECT_EQ(g.degree(v), 2);
    }
}

TEST(CsrBuilder, BeginEnd) {
    std::vector<int32_t> src = {0, 1};
    std::vector<int32_t> dst = {2, 2};
    CsrGraph g = build_csr_undirected(3, src, dst);

    // Vertex 2 has degree 2, vertices 0 and 1 have degree 1
    EXPECT_EQ(g.degree(0), 1);
    EXPECT_EQ(g.degree(1), 1);
    EXPECT_EQ(g.degree(2), 2);
}
