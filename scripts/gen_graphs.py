#!/usr/bin/env python3
"""Generate test graphs in edge-list format."""

import argparse
import random


def gen_grid(n: int, path: str) -> None:
    """Generate n×n grid graph."""
    edges = []
    for i in range(n):
        for j in range(n):
            v = i * n + j
            if i + 1 < n:
                edges.append((v, (i + 1) * n + j))
            if j + 1 < n:
                edges.append((v, i * n + (j + 1)))
    _write(path, n * n, edges)


def gen_random(n: int, m: int, path: str, seed: int = 42) -> None:
    """Generate random graph with n vertices and m edges."""
    rng = random.Random(seed)
    edges = set()
    while len(edges) < m:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edges.add((u, v))
    _write(path, n, list(edges))


def gen_complete(n: int, path: str) -> None:
    """Generate complete graph K_n."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    _write(path, n, edges)


def _write(path: str, n: int, edges: list) -> None:
    with open(path, "w") as f:
        f.write(f"{n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")
    print(f"Wrote {path}: |V|={n}, |E|={len(edges)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, help="Generate n×n grid")
    ap.add_argument("--random", nargs=2, type=int, metavar=("N", "M"),
                    help="Generate random graph with N vertices, M edges")
    ap.add_argument("--complete", type=int, metavar="N",
                    help="Generate complete graph K_n")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--output", default="graph.txt")
    args = ap.parse_args()

    if args.grid:
        gen_grid(args.grid, args.output)
    elif args.random:
        gen_random(args.random[0], args.random[1], args.output, args.seed)
    elif args.complete:
        gen_complete(args.complete, args.output)
    else:
        ap.print_help()
