#include <gtest/gtest.h>
#include "cuda_girth/girth_engine.h"
#include <vector>

using namespace cuda_girth;

// ---------------------------------------------------------------------------
// Small-graph correctness tests
// ---------------------------------------------------------------------------

TEST(Girth, Triangle) {
    // Girth of triangle = 3
    std::vector<int32_t> src = {0, 1, 2};
    std::vector<int32_t> dst = {1, 2, 0};
    auto g = build_csr_undirected(3, src, dst);

    EXPECT_EQ(girth_single_source(g, 0), 3);
    EXPECT_EQ(girth_all_sources(g), 3);
}

TEST(Girth, Square) {
    // Girth of square = 4
    std::vector<int32_t> src = {0, 1, 2, 3};
    std::vector<int32_t> dst = {1, 2, 3, 0};
    auto g = build_csr_undirected(4, src, dst);

    EXPECT_EQ(girth_all_sources(g), 4);
}

TEST(Girth, Petersen) {
    // Petersen graph: girth = 5
    // 10 vertices, 15 edges, vertex 0 connected to 1,4,5
    std::vector<int32_t> src = {
        0,0,0, 1,1,2, 2,3,3,4, 5,5,6, 6,7,
        1,4,5, 2,6,3, 7,4,8,9, 7,8,9, 8,9
    };
    std::vector<int32_t> dst = {
        1,4,5, 2,6,3, 7,4,8,9, 7,8,9, 8,9,
        0,0,0, 1,1,2, 2,3,3,4, 5,5,6, 6,7
    };
    auto g = build_csr_undirected(10, src, dst);

    int32_t g_val = girth_all_sources(g);
    EXPECT_EQ(g_val, 5);
}

TEST(Girth, Tree) {
    // A tree has no cycle → girth = 0
    std::vector<int32_t> src = {0, 0, 1, 1};
    std::vector<int32_t> dst = {1, 2, 3, 4};
    auto g = build_csr_undirected(5, src, dst);

    EXPECT_EQ(girth_all_sources(g), 0);
}

TEST(Girth, PerSource) {
    std::vector<int32_t> src = {0, 1, 2};
    std::vector<int32_t> dst = {1, 2, 0};
    auto g = build_csr_undirected(3, src, dst);

    std::vector<int32_t> out(3);
    girth_per_source(g, out.data());
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(out[i], 3);
    }
}
