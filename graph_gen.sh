#!/usr/bin/env bash
set -e  # exit on any error

RETRY_FAILS=false
MAX_SIZE_KB=150  # Default: skip binaries larger than 150KB

# Parse optional flags
while [[ "$1" == --* ]]; do
    case "$1" in
        --retry-fails)
            RETRY_FAILS=true
            shift
            ;;
        --max-size)
            MAX_SIZE_KB="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Usage check
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 [OPTIONS] <BIN_DIR> <DATA_DIR> <LOG_FILE>"
    echo "Example: $0 /path/to/binaries /path/to/output ./progress/O0.txt"
    echo ""
    echo "Options:"
    echo "  --retry-fails       Retry previously failed binaries"
    echo "  --max-size <KB>     Skip binaries larger than KB (default: 150)"
    exit 1
fi

BIN_DIR="$1"
DATA_DIR="$2"
LOG_FILE="$3"
FAIL_LOG="${LOG_FILE}.failed"
SKIP_LOG="${LOG_FILE}.skipped"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Touch log files if they don't exist
touch "$LOG_FILE"
touch "$FAIL_LOG"
touch "$SKIP_LOG"

echo "Max binary size: ${MAX_SIZE_KB}KB"

# Load successfully processed binaries
declare -A PROCESSED
while IFS= read -r line; do
    [ -n "$line" ] && PROCESSED["$line"]=1
done < "$LOG_FILE"

# Load failed binaries (skip them by default, unless --retry-fails)
declare -A FAILED
while IFS= read -r line; do
    [ -n "$line" ] && FAILED["$line"]=1
done < "$FAIL_LOG"

echo "Loaded ${#PROCESSED[@]} successful, ${#FAILED[@]} failed from logs"

if [ "$RETRY_FAILS" = true ]; then
    echo "Retrying failed binaries..."
    # Clear the failed log - we'll repopulate with any new failures
    > "$FAIL_LOG"
else
    echo "Skipping failed binaries (use --retry-fails to retry them)"
fi

# Count for progress
TOTAL=$(find "$BIN_DIR" -maxdepth 1 -type f | wc -l)
CURRENT=0
SKIPPED=0

# Loop over each binary
for BIN in "$BIN_DIR"/*; do
    [ -f "$BIN" ] || continue  # skip if not a file

    BASENAME=$(basename "$BIN")
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

    # Skip if too large
    FILE_SIZE_KB=$(( $(stat -c%s "$BIN") / 1024 ))
    if [ "$FILE_SIZE_KB" -gt "$MAX_SIZE_KB" ]; then
        ((SKIPPED++)) || true
        echo "[$CURRENT/$TOTAL] Skipping $BASENAME (${FILE_SIZE_KB}KB > ${MAX_SIZE_KB}KB)"
        # Log skipped file (only if not already logged)
        grep -qxF "$BASENAME" "$SKIP_LOG" 2>/dev/null || echo "$BASENAME" >> "$SKIP_LOG"
        continue
    fi

    DATASET_FILE="$DATA_DIR/${BASENAME}.pkl"
    echo "[$CURRENT/$TOTAL] Generating dataset for $BASENAME (${FILE_SIZE_KB}KB) -> $DATASET_FILE"

    if ./TYGR datagen "$BIN" "$DATASET_FILE"; then
        # Write to success log
        echo "$BASENAME" >> "$LOG_FILE"
        echo "[$CURRENT/$TOTAL] Done: $BASENAME"
    else
        # Write to failure log
        echo "$BASENAME" >> "$FAIL_LOG"
        echo "[$CURRENT/$TOTAL] FAILED: $BASENAME (logged to ${FAIL_LOG})"
    fi
done

# Count final stats
FAIL_COUNT=$(wc -l < "$FAIL_LOG" | tr -d ' ')
SKIP_SIZE_COUNT=$(wc -l < "$SKIP_LOG" | tr -d ' ')

echo ""
echo "=== Summary ==="
echo "Total binaries: $TOTAL"
echo "Processed: $((CURRENT - SKIPPED))"
echo "Skipped (already done/failed): $SKIPPED"
echo "Skipped (too large): $SKIP_SIZE_COUNT (see ${SKIP_LOG})"
echo "Failed: $FAIL_COUNT (see ${FAIL_LOG})"

