#!/usr/bin/env bash
set -e  # exit on any error

RETRY_FAILS=false

# Parse optional flags
while [[ "$1" == --* ]]; do
    case "$1" in
        --retry-fails)
            RETRY_FAILS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Usage check
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 [--retry-fails] <DATA_DIR> <MERGE_DIR> <BATCH_SIZE> <LOG_FILE>"
    echo "Example: $0 /path/to/pkls /path/to/merged 10 ./progress/merge.txt"
    echo ""
    echo "Options:"
    echo "  --retry-fails    Retry previously failed pkl files"
    exit 1
fi

DATA_DIR="$1"
MERGE_DIR="$2"
BATCH_SIZE="$3"
LOG_FILE="$4"
FAIL_LOG="${LOG_FILE}.failed"

MERGED_FILE="$MERGE_DIR/merged.pkl"
TMP_FILE="$MERGE_DIR/merged_tmp.pkl"

# Create directories if they don't exist
mkdir -p "$MERGE_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Touch log files if they don't exist
touch "$LOG_FILE"
touch "$FAIL_LOG"

# Load already merged pkl files
declare -A PROCESSED
while IFS= read -r line; do
    [ -n "$line" ] && PROCESSED["$line"]=1
done < "$LOG_FILE"

# Load failed pkl files
declare -A FAILED
while IFS= read -r line; do
    [ -n "$line" ] && FAILED["$line"]=1
done < "$FAIL_LOG"

echo "Loaded ${#PROCESSED[@]} merged, ${#FAILED[@]} failed from logs"

if [ "$RETRY_FAILS" = true ]; then
    echo "Retrying failed pkl files..."
    > "$FAIL_LOG"
else
    echo "Skipping failed pkl files (use --retry-fails to retry them)"
fi

# Collect pkl files to process
TO_PROCESS=()
for PKL in "$DATA_DIR"/*.pkl; do
    [ -f "$PKL" ] || continue
    BASENAME=$(basename "$PKL")

    # Skip if already merged
    [ "${PROCESSED[$BASENAME]}" ] && continue

    # Skip if failed (unless --retry-fails)
    if [ "$RETRY_FAILS" = false ] && [ "${FAILED[$BASENAME]}" ]; then
        continue
    fi

    TO_PROCESS+=("$PKL")
done

TOTAL=${#TO_PROCESS[@]}
echo "Found $TOTAL pkl files to merge (batch size: $BATCH_SIZE)"

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to merge."
    exit 0
fi

# Process in batches
BATCH_NUM=0
for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
    ((BATCH_NUM++)) || true

    # Get current batch
    BATCH=("${TO_PROCESS[@]:i:BATCH_SIZE}")
    BATCH_COUNT=${#BATCH[@]}

    echo ""
    echo "=== Batch $BATCH_NUM: Merging $BATCH_COUNT files ($(( i + 1 ))-$(( i + BATCH_COUNT ))/$TOTAL) ==="

    # Build merge command
    MERGE_ARGS=()

    # If merged.pkl exists, include it first
    if [ -f "$MERGED_FILE" ]; then
        MERGE_ARGS+=("$MERGED_FILE")
        echo "Including existing merged.pkl"
    fi

    # Add batch files
    for PKL in "${BATCH[@]}"; do
        MERGE_ARGS+=("$PKL")
        echo "  + $(basename "$PKL")"
    done

    echo "Merging to temporary file..."

    if ./TYGR datamerge "${MERGE_ARGS[@]}" -o "$TMP_FILE"; then
        # Success: replace merged.pkl with tmp
        mv "$TMP_FILE" "$MERGED_FILE"

        # Log all files in this batch as processed
        for PKL in "${BATCH[@]}"; do
            echo "$(basename "$PKL")" >> "$LOG_FILE"
        done

        echo "Batch $BATCH_NUM complete. Updated merged.pkl"
    else
        # Failure: log all files in batch as failed, cleanup tmp if exists
        rm -f "$TMP_FILE"

        for PKL in "${BATCH[@]}"; do
            echo "$(basename "$PKL")" >> "$FAIL_LOG"
        done

        echo "Batch $BATCH_NUM FAILED. Files logged to ${FAIL_LOG}"
        echo "Stopping to prevent data corruption. Fix issues and re-run."
        exit 1
    fi
done

# Count final stats
SUCCESS_COUNT=$(wc -l < "$LOG_FILE" | tr -d ' ')
FAIL_COUNT=$(wc -l < "$FAIL_LOG" | tr -d ' ')

echo ""
echo "=== Merge Complete ==="
echo "Total merged: $SUCCESS_COUNT"
echo "Total failed: $FAIL_COUNT"
echo "Output: $MERGED_FILE"
