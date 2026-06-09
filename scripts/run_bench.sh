#!/bin/bash
set -euo pipefail

BIN="${1:-./build/bench_girth}"
GRAPH_DIR="./test/graphs"

echo "=== Small Graph Benchmarks ==="

for g in triangle square petersen tree5; do
    if [ -f "$GRAPH_DIR/$g.txt" ]; then
        echo ""
        echo "--- $g ---"
        "$BIN" "$GRAPH_DIR/$g.txt" 2 5
    fi
done

echo ""
echo "=== Done ==="
