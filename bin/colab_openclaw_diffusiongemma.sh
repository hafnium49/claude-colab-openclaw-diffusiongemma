#!/usr/bin/env bash
set -euo pipefail

SESSION="openclaw-dg"
GPU="L4"
CONFIG="configs/diffusiongemma_nvfp4.json"
TASK="examples/prompt_task.json"
OUT_DIR="./runs/openclaw-dg"
KEEP_SESSION=0
REMOTE_SCRIPT="remote/remote_colab_openclaw_diffusiongemma.py"
STUB_SCRIPT="remote/colab_exec_stub.py"

usage() {
  cat <<'USAGE'
Usage:
  bash bin/colab_openclaw_diffusiongemma.sh [options]

Options:
  --session NAME       Colab CLI session name. Default: openclaw-dg
  --gpu GPU           GPU request, e.g. T4, L4, A100, H100. Default: L4
  --config PATH       Local config JSON. Default: configs/diffusiongemma_nvfp4.json
  --task PATH         Local prompt task JSON. Default: examples/prompt_task.json
  --out DIR           Local output directory. Default: ./runs/openclaw-dg
  --keep-session      Do not stop the Colab session after download
  -h, --help          Show help

Example:
  bash bin/colab_openclaw_diffusiongemma.sh \
    --session openclaw-dg \
    --gpu L4 \
    --config configs/diffusiongemma_nvfp4.json \
    --task examples/prompt_task.json \
    --out ./runs/openclaw-dg
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --keep-session) KEEP_SESSION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/local_colab_cli.log"
: > "$LOG"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }
}

run() {
  echo "+ $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

upload_control() {
  local action="$1"
  local tmp
  tmp=$(mktemp)
  cat > "$tmp" <<JSON
{"action":"$action"}
JSON
  run colab upload -s "$SESSION" "$tmp" /content/ocdg_control.json
  rm -f "$tmp"
}

exec_remote() {
  # `colab exec` defaults to a 30s code-execution timeout, but the bootstrap
  # phase (pip-install vLLM, download + load the model, start the gateway) runs
  # for many minutes. Without a generous --timeout the websocket is cut off
  # mid-bootstrap ("Connection was lost"). The remote script enforces its own
  # per-step timeouts, so this is just a safe upper bound on the whole phase.
  run colab exec -s "$SESSION" -f "$STUB_SCRIPT" --timeout "$COLAB_EXEC_TIMEOUT"
}

need colab
need python

# The colab CLI authenticates on every invocation, so route all calls through
# one wrapper to keep the strategy consistent. Default to ADC (works headlessly
# from existing gcloud application-default credentials); override with
# COLAB_AUTH=oauth2 if you have a browser-based OAuth client config.
COLAB_AUTH="${COLAB_AUTH:-adc}"
# Isolate this run's session state in a per-run scratch file so a concurrent
# `colab` command (e.g. a status check from another shell) can't race on the
# shared default state and prune this run's live session. The keep-alive daemon
# inherits --auth and --config automatically.
COLAB_CONFIG="${COLAB_CONFIG:-$OUT_DIR/colab_session_state.json}"
colab() { command colab --auth="$COLAB_AUTH" --config "$COLAB_CONFIG" "$@"; }

# Tear the session down on ANY exit (success or a set -e abort mid-phase) unless
# --keep-session was given, so a failed phase can't leak a billable VM.
cleanup() {
  if [[ "$KEEP_SESSION" -eq 0 ]]; then
    colab stop -s "$SESSION" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Upper bound (seconds) for each `colab exec`; must exceed the slowest phase
# (DiffusionGemma's install + weight download + load). Override per run if needed.
COLAB_EXEC_TIMEOUT="${COLAB_EXEC_TIMEOUT:-7200}"

python scripts/self_test.py | tee -a "$LOG"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2; exit 1
fi
if [[ ! -f "$TASK" ]]; then
  echo "Task not found: $TASK" >&2; exit 1
fi

run colab new -s "$SESSION" --gpu "$GPU"
run colab status -s "$SESSION"
run colab upload -s "$SESSION" "$REMOTE_SCRIPT" /content/remote_colab_openclaw_diffusiongemma.py
run colab upload -s "$SESSION" "$CONFIG" /content/ocdg_config.json
run colab upload -s "$SESSION" "$TASK" /content/ocdg_task.json

upload_control bootstrap
exec_remote

upload_control prompt
exec_remote

upload_control bundle
exec_remote

run colab ls -s "$SESSION" /content/ocdg_results || true
run colab download -s "$SESSION" /content/openclaw_diffusiongemma_results.zip "$OUT_DIR/openclaw_diffusiongemma_results.zip" || true
run colab download -s "$SESSION" /content/ocdg_results/manifest.json "$OUT_DIR/manifest.json" || true
run colab log -s "$SESSION" -o "$OUT_DIR/colab_session_log.ipynb" || true

if [[ "$KEEP_SESSION" -eq 0 ]]; then
  run colab stop -s "$SESSION" || true
else
  echo "Keeping Colab session: $SESSION" | tee -a "$LOG"
fi

echo "Artifacts written to: $OUT_DIR"
