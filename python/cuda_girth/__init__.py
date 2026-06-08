"""
cuda-girth: High-performance exact girth computation for large-scale graphs.

API
----
girth(G, source=None)  → int
    Compute exact girth of an undirected simple graph.

girth_per_source(G)    → dict[int, int]
    Per-source local girth.

validate_graph(G)      → None
    Check graph validity, raise on unsupported input.
"""

import networkx as nx

from .bridge import nx_to_csr
from . import _core

__all__ = ["girth", "girth_per_source", "validate_graph", "CsrGraph"]


class CsrGraph:
    """Pre-built CSR for repeated girth queries (avoids re-conversion)."""

    def __init__(self, G: nx.Graph):
        validate_graph(G)
        self._indptr, self._indices = nx_to_csr(G)
        self._num_vertices = G.number_of_nodes()

    @property
    def n(self) -> int:
        return self._num_vertices

    def girth(self, source: int | None = None) -> int:
        if source is not None:
            return _core.compute_girth_single(self._indptr, self._indices, source)
        return _core.compute_girth_all(self._indptr, self._indices)

    def girth_per_source(self) -> "dict[int, int]":
        out = _core.compute_girth_per_source(self._indptr, self._indices)
        return {s: out[s] for s in range(self._num_vertices) if out[s] > 0}


def girth(G: nx.Graph, source: int | None = None) -> int:
    """
    Compute exact girth of an undirected simple graph.

    Parameters
    ----------
    G : nx.Graph
        Undirected simple graph (no self-loops, no multi-edges).
    source : int, optional
        If given, compute local girth from this vertex only.

    Returns
    -------
    int
        The girth (length of shortest cycle). Returns 0 if the graph
        (or the source's component) is a forest.
    """
    csr = CsrGraph(G)
    return csr.girth(source)


def girth_per_source(G: nx.Graph) -> "dict[int, int]":
    """
    Compute per-source local girth.

    Returns a dict mapping source vertex → local girth.
    Vertices in tree components are omitted.
    """
    csr = CsrGraph(G)
    return csr.girth_per_source()


def validate_graph(G: nx.Graph) -> None:
    """Raise ValueError if G is not a valid input graph."""
    if G.is_directed():
        raise ValueError("Directed graphs are not supported")
    if nx.number_of_selfloops(G) > 0:
        raise ValueError("Self-loops are not supported")
    if G.is_multigraph():
        raise ValueError("Multi-graphs are not supported")
