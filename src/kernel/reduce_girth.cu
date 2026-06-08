/// reduce_girth.cu — multi-source girth reduction.
///
/// After computing per-source girth values (stored in unified memory),
/// a device-side parallel reduction finds the global minimum.
/// Uses CUB for efficient block-level reduction.

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_girth {

// ---------------------------------------------------------------------------
// Simple block-level min reduction over int32 array.
// For production, replace with cub::DeviceReduce::Min.
// ---------------------------------------------------------------------------
__global__ void block_reduce_min_kernel(
    const int32_t* __restrict__ values,
    int32_t         num_values,
    int32_t* __restrict__ block_mins   // size = num_blocks
) {
    extern __shared__ int32_t smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load into shared memory
    smem[tid] = (gid < num_values) ? values[gid] : INT32_MAX;
    __syncthreads();

    // Block-level reduction (power-of-2 unroll)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = min(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    // Write block result (exclude INT32_MAX sentinel)
    if (tid == 0) {
        block_mins[blockIdx.x] = (smem[0] > 0) ? smem[0] : INT32_MAX;
    }
}

void launch_reduce_per_source(const int32_t* values, int32_t num_values,
                               int32_t* result, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    int grid = (num_values + BLOCK - 1) / BLOCK;

    int32_t* block_mins;
    cudaMalloc(&block_mins, grid * sizeof(int32_t));

    block_reduce_min_kernel<<<grid, BLOCK, BLOCK * sizeof(int32_t), stream>>>(
        values, num_values, block_mins);

    // Final reduction of block_mins (grid is small, use a single block)
    block_reduce_min_kernel<<<1, BLOCK, BLOCK * sizeof(int32_t), stream>>>(
        block_mins, grid, result);

    cudaFree(block_mins);
}

}  // namespace cuda_girth
