#pragma once
#include <string>
#include "cuda_girth/csr_graph.h"

namespace cuda_girth {

/// Read edge-list file: first line "n m", then m lines "u v" (0-based).
CsrGraph read_edge_list(const std::string& path);

/// Write CSR as edge-list text (for debugging).
void write_csr_text(const CsrGraph& graph, const std::string& path);

}  // namespace cuda_girth
