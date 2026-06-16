#!/usr/bin/env bash
set -euo pipefail

# Defaults target the VALIDATED, fee-free path: llama.cpp serves Qwen3.5-9B (4-bit GGUF) on a
# Colab T4, OpenClaw points at it on loopback. For the original DiffusionGemma target, pass
# `--gpu L4 --config configs/diffusiongemma_nvfp4.json` (needs an L4 entitlement).
SESSION="openclaw-dg"
GPU="T4"
CONFIG="configs/llama_qwen9b.json"
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
  --gpu GPU           GPU request, e.g. T4, L4, A100, H100. Default: T4
  --config PATH       Local config JSON. Default: configs/llama_qwen9b.json
  --task PATH         Local task JSON. Default: examples/prompt_task.json
                      (task "mode":"research" runs the detached autonomous task instead of a
                       single prompt — see examples/research_task.json)
  --out DIR           Local output directory. Default: ./runs/openclaw-dg
  --keep-session      Do not stop the Colab session after download
  -h, --help          Show help

Examples:
  # Validated llama.cpp / Qwen3.5-9B smoke (single prompt) on a T4:
  bash bin/colab_openclaw_diffusiongemma.sh --config configs/llama_qwen9b.json \
    --task examples/prompt_task.json --out ./runs/llama9b

  # Autonomous, human-free deep-research run (detached + polled):
  bash bin/colab_openclaw_diffusiongemma.sh --config configs/llama_qwen9b.json \
    --task examples/research_task.json --out ./runs/research

  # Cheap orchestration smoke (0.5B GGUF):
  bash bin/colab_openclaw_diffusiongemma.sh --config configs/llama_smoke.json --out ./runs/smoke

  # Original DiffusionGemma target (needs an L4 entitlement):
  bash bin/colab_openclaw_diffusiongemma.sh --gpu L4 \
    --config configs/diffusiongemma_nvfp4.json --out ./runs/openclaw-dg
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
  local action="$1" tmp
  tmp=$(mktemp)
  printf '{"action":"%s"}\n' "$action" > "$tmp"
  run colab upload -s "$SESSION" "$tmp" /content/ocdg_control.json
  rm -f "$tmp"
}

exec_remote() {
  # `colab exec` defaults to a short timeout; phases that aren't fast (a synchronous prompt) need
  # a generous upper bound. The detached worker phases (bootstrap/task) return immediately and
  # are polled separately, so this mostly bounds the prompt exec.
  run colab exec -s "$SESSION" -f "$STUB_SCRIPT" --timeout "$COLAB_EXEC_TIMEOUT"
}

# Generic short-exec poller for a detached worker. $1=action (bootstrap_status|task_status),
# $2=STATE token (BOOTSTRAP_STATE|TASK_STATE), $3=budget seconds, $4=label. Must run under
# `set +e` (a dropped poll exec is expected and simply retried).
poll_worker() {
  local action="$1" token="$2" budget="$3" label="$4"
  local start=$SECONDS out state tmp
  tmp=$(mktemp); printf '{"action":"%s"}\n' "$action" > "$tmp"
  echo "[$label] waiting up to ${budget}s" | tee -a "$LOG"
  while (( SECONDS - start < budget )); do
    sleep 18
    colab upload -s "$SESSION" "$tmp" /content/ocdg_control.json >/dev/null 2>&1
    out=$(colab exec -s "$SESSION" -f "$STUB_SCRIPT" --timeout 90 2>&1)
    printf '%s\n' "$out" >> "$LOG"
    state=$(grep -o "${token}=[a-z]*" <<<"$out" | head -1)
    echo "[$label] +$(( SECONDS - start ))s ${state:-no-status}" | tee -a "$LOG"
    case "$out" in
      *${token}=ready*)  rm -f "$tmp"; echo "[$label] READY" | tee -a "$LOG"; return 0 ;;
      *${token}=failed*) rm -f "$tmp"; echo "[$label] FAILED" | tee -a "$LOG"; return 1 ;;
    esac
  done
  rm -f "$tmp"; echo "[$label] budget exhausted" | tee -a "$LOG"; return 1
}

need colab
need python

# Default to ADC (works headlessly from gcloud application-default credentials); override with
# COLAB_AUTH=oauth2. Isolate this run's session state in a per-run scratch file so a concurrent
# `colab` command can't race on the shared default state and prune this run's live session.
COLAB_AUTH="${COLAB_AUTH:-adc}"
COLAB_CONFIG="${COLAB_CONFIG:-$OUT_DIR/colab_session_state.json}"
colab() { command colab --auth="$COLAB_AUTH" --config "$COLAB_CONFIG" "$@"; }

# Tear the session down on ANY exit unless --keep-session, so a failed phase can't leak a billable VM.
cleanup() {
  if [[ "$KEEP_SESSION" -eq 0 ]]; then
    colab stop -s "$SESSION" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Upper bound (seconds) for a synchronous `colab exec` (mainly the prompt phase).
COLAB_EXEC_TIMEOUT="${COLAB_EXEC_TIMEOUT:-7200}"
# How long to poll for serve+onboard readiness. DERIVED FROM THE CONFIG's own step timeouts so it
# always OUTLASTS the detached worker's sequential worst case to bootstrap.done (wheel/vLLM install
# + the huggingface_hub install + GGUF/weight download + model load), plus margin — mirroring how
# TASK_BUDGET is derived. Frequent short polls keep the VM alive, so a generous backstop is cheap
# (the poller breaks the moment the worker reports ready/failed). Override with BOOTSTRAP_BUDGET=...
BOOTSTRAP_DEFAULT=$(python -c "
import json,sys
c=json.load(open(sys.argv[1]))
s=c.get('serve') or {}
backend=s.get('backend', 'vllm' if 'vllm' in c else 'llama_cpp')
sub=(s.get(backend) if s else c.get('vllm',{})) or {}
startup=int((s or c.get('vllm',{})).get('startup_timeout_seconds', 1800))
inst=int(sub.get('install_timeout_seconds', 1800 if backend=='llama_cpp' else 3600))
dl=int(sub.get('download_timeout_seconds', 1800))
hf=600 if backend=='llama_cpp' else 0
print(inst+hf+dl+startup+600)
" "$CONFIG" 2>/dev/null || echo 5400)
BOOTSTRAP_BUDGET="${BOOTSTRAP_BUDGET:-$BOOTSTRAP_DEFAULT}"

python scripts/self_test.py | tee -a "$LOG"

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$TASK" ]]   || { echo "Task not found: $TASK" >&2; exit 1; }

# Task mode decides the second phase: a single prompt, or the detached autonomous task.
MODE=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('mode','prompt'))" "$TASK" 2>/dev/null || echo prompt)
# Task poll budget must OUTLAST the worker: it caps each step at step_timeout_seconds (or
# max(120, total//steps)) and runs them sequentially, so worst case is per_step*num_steps; add
# margin. Mirrors the per_step formula in remote _task_run so the launcher never gives up first.
TASK_BUDGET=$(python -c "
import json,sys
t=json.load(open(sys.argv[1]))
steps=t.get('steps') or [t.get('prompt') or t.get('topic') or 'x']
total=int(t.get('timeout_seconds',1800))
per=int(t.get('step_timeout_seconds', max(120, total//max(1,len(steps)))))
print(per*len(steps)+600)
" "$TASK" 2>/dev/null || echo 3000)
# Single-prompt poll budget: the infer timeout + margin.
PROMPT_BUDGET=$(python -c "import json,sys; print(int(json.load(open(sys.argv[1])).get('timeout_seconds',900))+300)" "$TASK" 2>/dev/null || echo 1200)

run colab new -s "$SESSION" --gpu "$GPU"
run colab status -s "$SESSION"
run colab upload -s "$SESSION" "$REMOTE_SCRIPT" /content/remote_colab_openclaw_diffusiongemma.py
run colab upload -s "$SESSION" "$CONFIG" /content/ocdg_config.json
run colab upload -s "$SESSION" "$TASK" /content/ocdg_task.json

# Phases are best-effort from here: always reach the download + teardown below even if a phase
# fails or the runtime is reclaimed mid-flight.
set +e

# 1) Bootstrap: serve the backend + onboard OpenClaw, DETACHED, then poll until ready.
upload_control bootstrap
exec_remote
if poll_worker bootstrap_status BOOTSTRAP_STATE "$BOOTSTRAP_BUDGET" bootstrap; then BOOT_OK=1; else BOOT_OK=0; fi

# 2) Only infer if the backend is actually serving (skip the wasted exec otherwise). Each inference
#    phase is DETACHED + polled too, so no long synchronous exec can hit the ~10.5-min websocket drop.
if [[ "$BOOT_OK" -eq 1 ]]; then
  case "$MODE" in
    research|task|autonomous)
      echo "[mode] autonomous task (mode=$MODE)" | tee -a "$LOG"
      upload_control task; exec_remote
      poll_worker task_status TASK_STATE "$TASK_BUDGET" task
      ;;
    *)
      echo "[mode] single prompt (mode=$MODE)" | tee -a "$LOG"
      upload_control prompt; exec_remote
      poll_worker prompt_status PROMPT_STATE "$PROMPT_BUDGET" prompt
      ;;
  esac
else
  echo "[skip] bootstrap not ready — skipping inference, collecting diagnostics" | tee -a "$LOG"
fi

# 3) Bundle + collect.
upload_control bundle
exec_remote

run colab ls -s "$SESSION" /content/ocdg_results || true
run colab download -s "$SESSION" /content/openclaw_diffusiongemma_results.zip "$OUT_DIR/openclaw_diffusiongemma_results.zip" || true
run colab download -s "$SESSION" /content/ocdg_results/manifest.json "$OUT_DIR/manifest.json" || true
run colab download -s "$SESSION" /content/ocdg_results/research_result.md "$OUT_DIR/research_result.md" || true
run colab log -s "$SESSION" -o "$OUT_DIR/colab_session_log.ipynb" || true

if [[ "$KEEP_SESSION" -eq 0 ]]; then
  run colab stop -s "$SESSION" || true
else
  echo "Keeping Colab session: $SESSION" | tee -a "$LOG"
fi

echo "Artifacts written to: $OUT_DIR"
