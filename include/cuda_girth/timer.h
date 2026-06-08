#pragma once
#include <cstdio>
#include <string>
#include <chrono>
#include "cuda_girth/portability.h"
#include <cuda_runtime.h>

namespace cuda_girth {

/// RAII GPU timer — creates + destroys CUDA events.
struct GpuTimer {
    cudaEvent_t start_ev;
    cudaEvent_t stop_ev;
    float       elapsed_ms;

    GpuTimer() : elapsed_ms(0.0f) {
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_ev, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_ev, stream);
    }

    float sync() {
        cudaEventSynchronize(stop_ev);
        cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
        return elapsed_ms;
    }
};

/// Scoped CPU wall-clock timer for host sections.
struct CpuTimer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0;
    std::string label;

    explicit CpuTimer(std::string tag = "") : t0(Clock::now()), label(std::move(tag)) {}

    double elapsed_ms() const {
        auto t1 = Clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    ~CpuTimer() {
        if (!label.empty()) {
            fprintf(stderr, "[%s] %.3f ms\n", label.c_str(), elapsed_ms());
        }
    }
};

/// Check and clear CUDA error.
inline void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d — %s\n",
                file, line, cudaGetErrorString(err));
        exit(1);
    }
}
#define CUDA_CHECK(call) cuda_girth::check_cuda((call), __FILE__, __LINE__)

}  // namespace cuda_girth
