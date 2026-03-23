#!/usr/bin/env bash
set -e  # exit on any error

RETRY_FAILS=false
VERBOSE=""

# Parse optional flags
while [[ "$1" == --* ]]; do
    case "$1" in
        --retry-fails)
            RETRY_FAILS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Usage check
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 [OPTIONS] <INPUT_DIR> <OUTPUT_DIR> <LOG_FILE>"
    echo ""
    echo "Converts regular graph pkl files to TRUE hypergraph format."
    echo "(Different from star-expansion hypergraph - this creates hyperedge_index for HypergraphConv)"
    echo ""
    echo "Example: $0 /path/to/graphs /path/to/true_hypergraphs ./progress/true_hyper_O0.txt"
    echo ""
    echo "Options:"
    echo "  --retry-fails       Retry previously failed conversions"
    echo "  -v, --verbose       Verbose output during conversion"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
LOG_FILE="$3"
FAIL_LOG="${LOG_FILE}.failed"

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Touch log files if they don't exist
touch "$LOG_FILE"
touch "$FAIL_LOG"

# Load successfully processed files
declare -A PROCESSED
while IFS= read -r line; do
    [ -n "$line" ] && PROCESSED["$line"]=1
done < "$LOG_FILE"

# Load failed files (skip them by default, unless --retry-fails)
declare -A FAILED
while IFS= read -r line; do
    [ -n "$line" ] && FAILED["$line"]=1
done < "$FAIL_LOG"

echo "Loaded ${#PROCESSED[@]} successful, ${#FAILED[@]} failed from logs"

if [ "$RETRY_FAILS" = true ]; then
    echo "Retrying failed conversions..."
    > "$FAIL_LOG"
else
    echo "Skipping failed files (use --retry-fails to retry them)"
fi

# Count for progress
TOTAL=$(find "$INPUT_DIR" -maxdepth 1 -name "*.pkl" -type f | wc -l)
CURRENT=0
SKIPPED=0

# Loop over each pkl file
for INPUT_PKL in "$INPUT_DIR"/*.pkl; do
    [ -f "$INPUT_PKL" ] || continue

    BASENAME=$(basename "$INPUT_PKL")
    ((CURRENT++)) || true

    # Skip if already successfully processed
    if [ "${PROCESSED[$BASENAME]}" ]; then
        ((SKIPPED++)) || true
        echo "[$CURRENT/$TOTAL] Skipping $BASENAME (already processed)"
        continue
    fi

    # Skip if previously failed (unless --retry-fails)
    if [ "$RETRY_FAILS" = false ] && [ "${FAILED[$BASENAME]}" ]; then
        ((SKIPPED++)) || true
        echo "[$CURRENT/$TOTAL] Skipping $BASENAME (previously failed)"
        continue
    fi

    OUTPUT_PKL="$OUTPUT_DIR/$BASENAME"
    echo "[$CURRENT/$TOTAL] Converting $BASENAME -> $OUTPUT_PKL"

    if ./TYGR true_hypergraph_convert "$INPUT_PKL" "$OUTPUT_PKL" $VERBOSE; then
        echo "$BASENAME" >> "$LOG_FILE"
        echo "[$CURRENT/$TOTAL] Done: $BASENAME"
    else
        echo "$BASENAME" >> "$FAIL_LOG"
        echo "[$CURRENT/$TOTAL] FAILED: $BASENAME (logged to ${FAIL_LOG})"
    fi
done

# Count final stats
SUCCESS_COUNT=$(wc -l < "$LOG_FILE" | tr -d ' ')
FAIL_COUNT=$(wc -l < "$FAIL_LOG" | tr -d ' ')

echo ""
echo "=== Summary ==="
echo "Total pkl files: $TOTAL"
echo "Converted: $((CURRENT - SKIPPED))"
echo "Skipped (already done/failed): $SKIPPED"
echo "Successfully converted: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT (see ${FAIL_LOG})"
