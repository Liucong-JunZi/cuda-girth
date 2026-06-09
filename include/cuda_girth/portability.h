#pragma once

// ---------------------------------------------------------------------------
// Portability: define __host__ / __device__ for non-CUDA tooling
// ---------------------------------------------------------------------------
#ifndef __CUDACC__
    #define __host__
    #define __device__
    #define __forceinline__ inline
#endif
