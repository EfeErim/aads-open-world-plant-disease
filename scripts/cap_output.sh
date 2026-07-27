#!/usr/bin/env bash
# Simple wrapper to run a command and cap its output to a fixed number of bytes.
# Usage: ./cap_output.sh <max_bytes> <command> [args...]

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <max_bytes> <command> [args...]" >&2
  exit 2
fi

MAX_BYTES="$1"
shift

# Run the command, capture stdout+stderr, and trim to MAX_BYTES
"$@" 2>&1 | head -c "$MAX_BYTES"
