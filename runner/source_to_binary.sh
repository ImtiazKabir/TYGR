#!/usr/bin/env bash

set -euo pipefail

src_dir="${1:?Usage: $0 <src_dir> <bin_dir>}"
bin_dir="${2:?Usage: $0 <src_dir> <bin_dir>}"

mkdir -p "$bin_dir"

for f in "$src_dir"/*.c; do
    [ -e "$f" ] || continue  # handle no matches

    base="$(basename "$f" .c)"
    gcc -O0 -g -c "$f" -o "$bin_dir/$base.o"
done
