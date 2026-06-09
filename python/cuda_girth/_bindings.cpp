#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>
#include "cuda_girth/girth_engine.h"
#include "cuda_girth/csr_graph.h"

namespace py = pybind11;
namespace cg = cuda_girth;

// ---------------------------------------------------------------------------
// Convert NumPy int32 CSR arrays → CsrGraph on device, run girth, return int.
// ---------------------------------------------------------------------------

static cg::CsrGraph numpy_to_csr(py::array_t<int32_t> indptr,
                                  py::array_t<int32_t> indices) {
    auto buf_ptr = indptr.request();
    auto buf_ind = indices.request();

    int32_t n = static_cast<int32_t>(buf_ptr.size) - 1;
    size_t total_edges = buf_ind.size;

    std::vector<int32_t> row, col;
    row.reserve(total_edges);
    col.reserve(total_edges);

    for (int32_t v = 0; v < n; ++v) {
        int32_t start = static_cast<int32_t*>(buf_ptr.ptr)[v];
        int32_t end   = static_cast<int32_t*>(buf_ptr.ptr)[v + 1];
        for (int32_t e = start; e < end; ++e) {
            row.push_back(v);
            col.push_back(static_cast<int32_t*>(buf_ind.ptr)[e]);
        }
    }

    // CSR is already symmetric (undirected NetworkX graph).
    // is_symmetric=true avoids double-counting edges.
    return cg::build_csr_from_edges(row, col, n, /*is_symmetric=*/true);
}

static int32_t compute_girth_single(py::array_t<int32_t> indptr,
                                     py::array_t<int32_t> indices,
                                     int32_t source) {
    auto graph = numpy_to_csr(indptr, indices);
    return cg::girth_single_source(graph, source);
}

static int32_t compute_girth_all(py::array_t<int32_t> indptr,
                                  py::array_t<int32_t> indices) {
    auto graph = numpy_to_csr(indptr, indices);
    return cg::girth_all_sources(graph);
}

static py::array_t<int32_t> compute_girth_per_source(
    py::array_t<int32_t> indptr,
    py::array_t<int32_t> indices)
{
    auto graph = numpy_to_csr(indptr, indices);
    int32_t n = graph.num_vertices;

    auto result = py::array_t<int32_t>(n);
    auto buf = result.request();
    int32_t* out = static_cast<int32_t*>(buf.ptr);

    cg::girth_per_source(graph, out);
    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_core, m) {
    m.doc() = "CUDA-accelerated exact girth computation";

    m.def("compute_girth_single", &compute_girth_single,
          py::arg("indptr"), py::arg("indices"), py::arg("source"),
          "Compute girth from a single source vertex.");

    m.def("compute_girth_all", &compute_girth_all,
          py::arg("indptr"), py::arg("indices"),
          "Compute exact girth (all sources).");

    m.def("compute_girth_per_source", &compute_girth_per_source,
          py::arg("indptr"), py::arg("indices"),
          "Return per-source girth as int32 numpy array.");
}
