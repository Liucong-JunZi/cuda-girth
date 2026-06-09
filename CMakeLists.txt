cmake_minimum_required(VERSION 3.21)
project(cuda-girth VERSION 0.1.0 LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CUDA_GIRTH_BUILD_TESTS   "Build unit tests"   ON)
option(CUDA_GIRTH_BUILD_BENCH   "Build benchmarks"   OFF)
option(CUDA_GIRTH_BUILD_PYTHON  "Build Python bindings" ON)
option(CUDA_GIRTH_ENABLE_ITT    "Enable ITT/PTI profiling markers" OFF)

# ---------------------------------------------------------------------------
# CUDA architecture
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# -- pybind11 (optional)
if(CUDA_GIRTH_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# -- GTest (optional)
if(CUDA_GIRTH_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

# ---------------------------------------------------------------------------
# Core library (header-only host utilities + compiled device kernels)
# ---------------------------------------------------------------------------
add_library(cuda_girth_core STATIC
    src/host/csr_builder.cpp
    src/host/bfs_state.cpp
    src/host/graph_io.cpp
    src/host/girth_engine.cpp
    src/host/batch_engine.cpp
    src/kernel/bfs_level.cu
    src/kernel/frontier_ops.cu
    src/kernel/reduce_girth.cu
)
target_include_directories(cuda_girth_core
    PUBLIC
        include
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(cuda_girth_core
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
)
# Ensure object files are position-independent — needed when _core.so links this archive.
set_target_properties(cuda_girth_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# CLI executable
# ---------------------------------------------------------------------------
add_executable(girth_cli src/host/main.cpp)
target_link_libraries(girth_cli PRIVATE cuda_girth_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CUDA_GIRTH_BUILD_TESTS)
    enable_testing()
    add_executable(test_csr  test/test_csr.cpp)
    add_executable(test_bfs  test/test_bfs.cpp)
    add_executable(test_girth test/test_girth.cpp)
    foreach(tgt test_csr test_bfs test_girth)
        target_link_libraries(${tgt} PRIVATE cuda_girth_core GTest::gtest_main)
        add_test(NAME ${tgt} COMMAND ${tgt})
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(CUDA_GIRTH_BUILD_BENCH)
    add_executable(bench_girth bench/bench_girth.cpp)
    target_link_libraries(bench_girth PRIVATE cuda_girth_core)
endif()

# ---------------------------------------------------------------------------
# Python bindings
# ---------------------------------------------------------------------------
if(CUDA_GIRTH_BUILD_PYTHON)
    pybind11_add_module(_core python/cuda_girth/_bindings.cpp)
    target_link_libraries(_core PRIVATE cuda_girth_core)
    target_include_directories(_core PRIVATE include)
endif()
