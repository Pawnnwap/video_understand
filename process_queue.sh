#!/usr/bin/env sh
set -eu

pause_exit() {
  code="${1:-1}"
  shift || true
  if [ "$#" -gt 0 ]; then
    for line do
      echo "$line" >&2
    done
  fi
  echo >&2
  printf "Press Enter to exit..." >&2
  IFS= read -r _ || true
  exit "$code"
}

# Resolve this script's real directory so it works from any current directory.
SCRIPT_DIR=$(
  CDPATH= cd -- "$(dirname -- "$0")" >/dev/null 2>&1
  pwd -P
)

PYTHON="$SCRIPT_DIR/.venv/bin/python"
QUEUE="$SCRIPT_DIR/process_queue.py"

# GUI/double-click launches often have a minimal PATH. Resolve opencode here
# so the Python pipeline does not depend on an interactive shell profile.
if [ -n "${OPENCODE_BIN:-}" ]; then
  if [ -x "$OPENCODE_BIN" ]; then
    PATH="$(dirname -- "$OPENCODE_BIN"):${PATH:-}"
    export PATH OPENCODE_BIN
  else
    pause_exit 1 "OPENCODE_BIN is set but not executable: $OPENCODE_BIN"
  fi
else
  OPENCODE_CANDIDATES="${SCRIPT_DIR}/.opencode/bin/opencode
${HOME:-}/.opencode/bin/opencode
${HOME:-}/.local/bin/opencode
/usr/local/bin/opencode
/usr/bin/opencode"
  for candidate in $OPENCODE_CANDIDATES; do
    if [ -x "$candidate" ]; then
      OPENCODE_BIN="$candidate"
      PATH="$(dirname -- "$candidate"):${PATH:-}"
      export PATH OPENCODE_BIN
      break
    fi
  done
fi

if [ ! -x "$PYTHON" ]; then
  pause_exit 1 "Missing virtualenv Python: $PYTHON" "Run deployment/setup first, or create .venv in this repo."
fi

if [ ! -f "$QUEUE" ]; then
  pause_exit 1 "Missing queue runner: $QUEUE"
fi

cd "$SCRIPT_DIR"

if [ "$#" -eq 0 ]; then
  printf "Video codes / URLs / paths: "
  if ! IFS= read -r SOURCES; then
    pause_exit 1 "No sources provided."
  fi
  if [ -z "$SOURCES" ]; then
    pause_exit 1 "No sources provided."
  fi
  set -- $SOURCES
fi

set +e
"$PYTHON" "$QUEUE" "$@"
status=$?
set -e
if [ "$status" -ne 0 ]; then
  pause_exit "$status" "process_queue.py failed with exit code $status."
fi
