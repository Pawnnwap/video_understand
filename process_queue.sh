#!/usr/bin/env sh
set -eu

# Resolve this script's real directory so it works from any current directory.
SCRIPT_DIR=$(
  CDPATH= cd -- "$(dirname -- "$0")" >/dev/null 2>&1
  pwd -P
)

PYTHON="$SCRIPT_DIR/.venv/bin/python"
QUEUE="$SCRIPT_DIR/process_queue.py"

if [ ! -x "$PYTHON" ]; then
  echo "Missing virtualenv Python: $PYTHON" >&2
  echo "Run deployment/setup first, or create .venv in this repo." >&2
  exit 1
fi

if [ ! -f "$QUEUE" ]; then
  echo "Missing queue runner: $QUEUE" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

if [ "$#" -eq 0 ]; then
  printf "Video codes / URLs / paths: "
  if ! IFS= read -r SOURCES; then
    echo "No sources provided." >&2
    exit 1
  fi
  if [ -z "$SOURCES" ]; then
    echo "No sources provided." >&2
    exit 1
  fi
  set -- "$SOURCES"
fi

exec "$PYTHON" "$QUEUE" "$@"
