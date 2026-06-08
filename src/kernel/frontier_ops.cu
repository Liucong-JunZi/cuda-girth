/// frontier_ops.cu — frontier compaction and deduplication kernels.
///
/// During BFS, multiple warps may race to claim the same neighbour via
/// atomicCAS. The hardware resolves the race (exactly one warp wins),
/// but the losing warps may still append duplicate entries to
/// next_frontier[]. A post-expansion dedup pass eliminates these.
///
/// Also provides: frontier-to-bitmask conversion for sparse frontier
/// representation (useful for very large graphs with thin frontiers).

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_girth {

// ---------------------------------------------------------------------------
// Simple block-level dedup: sort + unique within each block's slice.
// For production, Thrust/CUB sort+unique would be more efficient.
// ---------------------------------------------------------------------------
__global__ void dedup_frontier_kernel(
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size,
    int32_t  max_size
) {
    // Stub: in practice, coordinate with CUB DeviceRadixSort + DeviceSelect::Unique
    // For small frontiers, block-level bitmask dedup is sufficient.
    (void)frontier;
    (void)frontier_size;
    (void)max_size;
}

// ---------------------------------------------------------------------------
// Convert compact frontier to dense bitmap for fast O(1) membership test.
// ---------------------------------------------------------------------------
__global__ void frontier_to_bitmap_kernel(
    const int32_t* __restrict__ frontier,
    int32_t         frontier_size,
    int32_t* __restrict__ bitmap       // size ceil(n/32) words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int v = frontier[tid];
    int word = v / 32;
    int bit  = v % 32;
    atomicOr(&bitmap[word], 1u << bit);
}

void launch_frontier_to_bitmap(const int32_t* frontier, int32_t size,
                                int32_t* bitmap, cudaStream_t stream) {
    int block = 256;
    int grid  = (size + block - 1) / block;
    frontier_to_bitmap_kernel<<<grid, block, 0, stream>>>(
        frontier, size, bitmap);
}

}  // namespace cuda_girth
