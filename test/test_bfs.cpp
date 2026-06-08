#include <gtest/gtest.h>
#include "cuda_girth/bfs_state.h"
#include "cuda_girth/csr_graph.h"
#include <vector>

using namespace cuda_girth;

TEST(BfsState, InitAndFree) {
    std::vector<int32_t> src = {0, 1};
    std::vector<int32_t> dst = {2, 2};
    auto g = build_csr_undirected(3, src, dst);

    BfsState state;
    state.init(/*source=*/0, g);

    EXPECT_EQ(state.source, 0);
    EXPECT_EQ(state.current_level, 0);
    EXPECT_EQ(state.frontier_size, 1);
    EXPECT_EQ(state.frontier[0], 0);
    EXPECT_EQ(state.levels[0], 0);

    // All other vertices unvisited
    for (int v = 1; v < 3; ++v) {
        EXPECT_EQ(state.levels[v], -1);
    }
}

TEST(BfsState, Move) {
    std::vector<int32_t> src = {0};
    std::vector<int32_t> dst = {1};
    auto g = build_csr_undirected(2, src, dst);

    BfsState s1;
    s1.init(0, g);
    int32_t* ptr = s1.levels;

    BfsState s2 = std::move(s1);
    EXPECT_EQ(s1.levels, nullptr);
    EXPECT_EQ(s2.levels, ptr);
    EXPECT_EQ(s2.source, 0);
}
