"""Bridge: NetworkX graph → NumPy CSR arrays for pybind11."""

from typing import Tuple
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix


def nx_to_csr(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a NetworkX undirected graph to CSR arrays.

    Returns (indptr, indices) as contiguous int32 NumPy arrays
    suitable for zero-copy handoff to CUDA via pybind11.

    The adjacency matrix is symmetrized automatically.
    """
    # to_scipy_sparse_array produces a symmetric CSR for undirected graphs.
    # Use int32 to match GPU kernel expectations.
    csr: csr_matrix = nx.to_scipy_sparse_array(
        G, dtype=np.int32, format="csr"
    )
    # Ensure contiguous (normally already true for fresh CSR)
    indptr  = np.ascontiguousarray(csr.indptr,  dtype=np.int32)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    return indptr, indices


def csr_to_nx(indptr: np.ndarray, indices: np.ndarray) -> nx.Graph:
    """
    Reconstruct a NetworkX graph from CSR arrays (for testing round-trip).
    """
    G = nx.Graph()
    n = len(indptr) - 1
    G.add_nodes_from(range(n))
    for u in range(n):
        for j in range(indptr[u], indptr[u + 1]):
            v = int(indices[j])
            if u < v:
                G.add_edge(u, v)
    return G
