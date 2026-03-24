#!/usr/bin/env bash
#
# Compare actual vs inferred types for a C source file
#
# Usage:
#   ./runner/compare_source.sh <source.c> <model> [-m method]
#
# Example:
#   ./runner/compare_source.sh prediction/hello.c models/rgat_on_star_hypergraph/rgat_on_star_hypergraph.best.model -m glow_rgat
#

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <source.c> <model> [-m method] [other options...]"
    echo ""
    echo "Examples:"
    echo "  $0 my_program.c models/rgcn.best.model"
    echo "  $0 my_program.c models/rgat.best.model -m glow_rgat"
    echo "  $0 my_program.c models/hgnn.best.model -m glow_hgnn"
    echo ""
    echo "Available methods:"
    echo "  glow       - RGCN (default)"
    echo "  glow_gat   - GAT"
    echo "  glow_rgat  - RGAT"
    echo "  glow_hgnn  - True Hypergraph GNN"
    exit 1
fi

SOURCE_FILE="$1"
MODEL="$2"
shift 2

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file not found: $SOURCE_FILE"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found: $MODEL"
    exit 1
fi

# Create temp directory for compiled binary
TEMP_DIR=$(mktemp -d)
BASENAME=$(basename "$SOURCE_FILE" .c)
BINARY="${TEMP_DIR}/${BASENAME}.o"

echo "=== TYGR Type Inference Comparison ==="
echo ""
echo "Source: $SOURCE_FILE"
echo "Model:  $MODEL"
echo ""

# Compile with debug info
echo "Compiling with debug info..."
gcc -g -O0 -c "$SOURCE_FILE" -o "$BINARY"
echo "Binary: $BINARY"
echo ""

# Run comparison
echo "Running type inference..."
echo ""
./TYGR compare_types "$MODEL" "$BINARY" "$@"

# Cleanup
rm -rf "$TEMP_DIR"
