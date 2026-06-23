#!/usr/bin/env python3
"""Remote Colab orchestrator for OpenClaw + a self-hosted LLM.

Executed inside a Google Colab VM by `colab exec` (via colab_exec_stub.py). Reads
control/config/task files from /content and writes artifacts to /content/ocdg_results,
then builds /content/openclaw_diffusiongemma_results.zip.

Serving backend is CONFIG-DRIVEN (`serve.backend`):
  - "llama_cpp" (VALIDATED, fee-free): prebuilt llama-cpp-python[server] CUDA wheel serves a
    local GGUF on loopback. The only backend that serves >=3B on a Colab T4 (vLLM's FlashInfer
    crashes on Turing/sm_75). See docs/t4_llama_cpp_serving.md.
  - "vllm": kept for the original DiffusionGemma/L4 target (old configs without a `serve` block
    fall back to this via their top-level `vllm` section).

EVERY heavy phase (bootstrap, prompt, task) runs DETACHED via `--worker` and is polled through a
short *_status exec, so no single `colab exec` is held open through a multi-minute step (which
would hit the ~10.5-min websocket-drop). Phases are dispatched by /content/ocdg_control.json
`{"action": ...}`, re-uploaded by the launcher before each exec.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE = Path('/content')
RESULTS = BASE / 'ocdg_results'
CONFIG_PATH = BASE / 'ocdg_config.json'
CONTROL_PATH = BASE / 'ocdg_control.json'
TASK_PATH = BASE / 'ocdg_task.json'
ZIP_PATH = BASE / 'openclaw_diffusiongemma_results.zip'
SELF_PATH = BASE / 'remote_colab_openclaw_diffusiongemma.py'
GGUF_DIR = BASE / 'gguf'
SECRETS_PATH = BASE / 'ocdg_secrets.json'  # launcher-forwarded secrets (allowlist), loaded into env

# Secrets the launcher may forward into the VM. ALLOWLIST — never apply anything outside this set,
# and NEVER the user's own OPENCLAW_GATEWAY_TOKEN (the Colab gateway mints its own loopback token).
FORWARDED_SECRET_KEYS = ('BRAVE_API_KEY',)

# (status_file, done_file) per detached phase.
STATUS_PATH = RESULTS / 'bootstrap_status.json'
DONE_PATH = RESULTS / 'bootstrap.done'
PROMPT_STATUS_PATH = RESULTS / 'prompt_status.json'
PROMPT_DONE_PATH = RESULTS / 'prompt.done'
TASK_STATUS_PATH = RESULTS / 'task_status.json'
TASK_DONE_PATH = RESULTS / 'task.done'

# Resolve the openclaw binary by PATH -> never 'openclaw: command not found' regardless of PATH
# quirks. (The npm installer symlinks it into the global bin, e.g. /usr/bin/openclaw.)
PATH_PREFIX = 'export PATH="$(npm prefix -g)/bin:$PATH"; '


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        if default is None:
            raise FileNotFoundError(path)
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')


def write_status(path: Path, stage: str, extra: Optional[Dict[str, Any]] = None) -> None:
    data: Dict[str, Any] = {'stage': stage, 'time': now()}
    if extra:
        data.update(extra)
    write_json(path, data)


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(text)
        if not text.endswith('\n'):
            f.write('\n')


def run(cmd: str, log_name: str, check: bool = False, env: Optional[Dict[str, str]] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    log_path = RESULTS / log_name
    append(log_path, f"\n[{now()}] $ {cmd}\n")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            cmd, shell=True, executable='/bin/bash',
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=merged_env, timeout=timeout,
        )
        rc, out = proc.returncode, proc.stdout
    except subprocess.TimeoutExpired as exc:
        # With text=True, TimeoutExpired.output can still come back as undecoded bytes (a CPython
        # quirk — communicate() on timeout returns the raw buffer), so decode defensively. Without
        # this the timeout path raises TypeError("can't concat str to bytes") and loses the whole
        # phase instead of recording rc=124 + partial output.
        partial = exc.output
        if isinstance(partial, (bytes, bytearray)):
            partial = partial.decode('utf-8', 'replace')
        rc, out = 124, (partial or '') + f"\n[timeout after {timeout}s]"
    append(log_path, out)
    result = {'cmd': cmd, 'returncode': rc, 'log': str(log_path)}
    if check and rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {cmd}\nSee {log_path}")
    return result


def http_get_json(url: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    # Auth header is harmless for llama.cpp (no api_key -> ignored) and satisfies vLLM if set.
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer vllm-local'})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode('utf-8'))


def wait_for_url(url: str, seconds: int, log_name: str) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            data = http_get_json(url, timeout_s=5)
            write_json(RESULTS / 'models.json', data)
            return True
        except Exception as exc:
            append(RESULTS / log_name, f"[{now()}] waiting for {url}: {exc}\n")
            time.sleep(5)
    return False


def extract_infer_text(raw: str) -> Optional[str]:
    """Salvage the model text from an `openclaw infer ... --json` log.

    The log also contains the echoed command (which embeds the prompt, possibly with braces) and
    other lines, so a naive first-{ to last-} slice is fragile. Use a real JSON parser
    (raw_decode is brace/quote aware) to scan every `{...}` object and keep the LAST one that has
    an `outputs` key — that is OpenClaw's result object.
    """
    dec = json.JSONDecoder()
    best: Optional[Dict[str, Any]] = None
    idx = raw.find('{')
    while idx != -1:
        try:
            obj, end = dec.raw_decode(raw, idx)
            if isinstance(obj, dict) and 'outputs' in obj:
                best = obj
            idx = raw.find('{', max(end, idx + 1))
        except json.JSONDecodeError:
            idx = raw.find('{', idx + 1)
    # Only count it as text when outputs is a non-empty list whose first entry has non-blank text.
    # An empty/incomplete completion (outputs:[] or outputs:[{}]) must read as "no text" so the
    # got_text success signal stays honest (don't dump the raw object as if it were the answer).
    if not isinstance(best, dict):
        return None
    outs = best.get('outputs')
    if isinstance(outs, list) and outs and isinstance(outs[0], dict):
        text = outs[0].get('text')
        if isinstance(text, str) and text.strip():
            return text
    return None


def extract_agent_text(raw: str) -> Optional[str]:
    """Salvage the reply text from an `openclaw agent ... --json` log.

    The agent JSON shape varies by version (payloads/metadata/deliveryStatus, or openai/infer-style
    fallbacks), and the log also echoes the command, so scan every {...} object (raw_decode is
    brace/quote aware) and keep the LAST one that yields non-blank text. Each object is checked in
    priority order (payloads first) and returns the FIRST populated field so a request echo doesn't
    get mistaken for the reply."""
    def _from_obj(obj: Any) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        pl = obj.get('payloads')
        if isinstance(pl, list):
            parts = []
            for it in pl:
                if isinstance(it, str) and it.strip():
                    parts.append(it)
                elif isinstance(it, dict):
                    for k in ('text', 'content', 'body'):
                        v = it.get(k)
                        if isinstance(v, str) and v.strip():
                            parts.append(v)
                            break
            if parts:
                return "\n".join(parts).strip()
        for k in ('reply', 'response', 'text', 'content', 'output', 'message'):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        ch = obj.get('choices')
        if isinstance(ch, list) and ch and isinstance(ch[0], dict):
            msg = ch[0].get('message') or {}
            if isinstance(msg, dict) and isinstance(msg.get('content'), str) and msg['content'].strip():
                return msg['content'].strip()
        outs = obj.get('outputs')
        if isinstance(outs, list) and outs and isinstance(outs[0], dict):
            t = outs[0].get('text')
            if isinstance(t, str) and t.strip():
                return t.strip()
        return None

    dec = json.JSONDecoder()
    best: Optional[str] = None
    idx = raw.find('{')
    while idx != -1:
        try:
            obj, end = dec.raw_decode(raw, idx)
            t = _from_obj(obj)
            if t:
                best = t
            idx = raw.find('{', max(end, idx + 1))
        except json.JSONDecodeError:
            idx = raw.find('{', idx + 1)
    return best


def load_forwarded_secrets() -> None:
    """Load launcher-forwarded secrets (ALLOWLIST only) into os.environ.

    The launcher uploads /content/ocdg_secrets.json with a minimal allowlist (e.g. BRAVE_API_KEY for
    web search). Every phase/worker is a fresh process, so we re-load here at each entrypoint. We
    apply ONLY keys in FORWARDED_SECRET_KEYS — never the user's OPENCLAW_GATEWAY_TOKEN — and the file
    lives OUTSIDE ocdg_results/ so it is never bundled into the downloaded zip. Best-effort: a
    missing/malformed file must not crash a phase.
    """
    try:
        if not SECRETS_PATH.exists():
            return
        data = json.loads(SECRETS_PATH.read_text(encoding='utf-8'))
        for key in FORWARDED_SECRET_KEYS:
            val = data.get(key)
            if isinstance(val, str) and val:
                os.environ[key] = val
    except Exception:
        pass


def oc_env(config: Dict[str, Any]) -> Dict[str, str]:
    """Resolve the OpenClaw gateway token + provider api key (+ forwarded web-search keys).

    Each `colab exec` is a fresh process, so later (detached) phases must re-derive these from
    the SAME source as configure_openclaw(): env first (a real Colab secret wins), then the
    config's openclaw.{gateway_token,vllm_api_key}, then a loopback default.
    """
    ocfg = config.get('openclaw', {})
    env = {
        'OPENCLAW_GATEWAY_TOKEN': os.environ.get('OPENCLAW_GATEWAY_TOKEN') or ocfg.get('gateway_token') or 'colab-openclaw-local-token',
        'VLLM_API_KEY': os.environ.get('VLLM_API_KEY') or ocfg.get('vllm_api_key') or 'vllm-local',
    }
    # Forward web-search provider keys (allowlist) into the OpenClaw/gateway PROCESS env so the brave
    # web_search plugin can read BRAVE_API_KEY (provider keys are blocked from workspace .env files).
    # Only keys actually present are added; NEVER the user's own gateway token (loopback default above).
    for key in FORWARDED_SECRET_KEYS:
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


# ---------------------------------------------------------------------------
# Serve-backend abstraction (config-driven: llama_cpp | vllm)
# ---------------------------------------------------------------------------

def serve_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize serving config across the new `serve` schema and the legacy `vllm` block."""
    s = config.get('serve')
    if s is not None:
        backend = s.get('backend', 'llama_cpp')
        return {
            'backend': backend,
            'host': s.get('host', '127.0.0.1'),
            'port': int(s.get('port', 8000)),
            'startup_timeout_seconds': int(s.get('startup_timeout_seconds', 900)),
            'install': s.get('install', s.get(backend, {}).get('install', True)),
            backend: s.get(backend, {}),
        }
    v = config.get('vllm', {})
    return {
        'backend': 'vllm',
        'host': v.get('host', '127.0.0.1'),
        'port': int(v.get('port', 8000)),
        'startup_timeout_seconds': int(v.get('startup_timeout_seconds', 1800)),
        'install': v.get('install', True),
        'vllm': v,
    }


def collect_environment() -> None:
    lines = [f"timestamp_utc: {now()}", f"python: {sys.version}"]
    for cmd in ['uname -a', 'nvidia-smi', 'python -m pip --version', 'df -h /content', 'free -h']:
        p = subprocess.run(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines.append(f"\n$ {cmd}\n{p.stdout}")
    (RESULTS / 'environment.txt').write_text('\n'.join(lines), encoding='utf-8')


def install_vllm(scfg: Dict[str, Any]) -> None:
    vcfg = scfg.get('vllm', {})
    if not scfg.get('install', True):
        return
    install_cmd = vcfg.get('install_command') or (
        'python -m pip install -U vllm --pre '
        '--extra-index-url https://wheels.vllm.ai/nightly/cu129 '
        '--extra-index-url https://download.pytorch.org/whl/cu129'
    )
    run('python -m pip install -U uv', 'install.log', check=False, timeout=900)
    run(install_cmd, 'install.log', check=True, timeout=int(vcfg.get('install_timeout_seconds', 3600)))


def start_vllm(config: Dict[str, Any], scfg: Dict[str, Any]) -> Dict[str, Any]:
    vcfg = scfg.get('vllm', {})
    model_id = config['model']['id']
    host, port = scfg['host'], scfg['port']
    max_wait = scfg['startup_timeout_seconds']

    env_exports = {
        'VLLM_USE_V2_MODEL_RUNNER': str(vcfg.get('use_v2_model_runner', '1')),
        'HF_HUB_ENABLE_HF_TRANSFER': str(vcfg.get('hf_transfer', '1')),
    }
    for tok in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
        if os.environ.get(tok):
            env_exports[tok] = os.environ[tok]

    args: List[str] = ['vllm', 'serve', shlex.quote(model_id)]
    # shlex.quote EACH serve arg — diffusion configs pass JSON values (e.g. --hf-overrides
    # '{"diffusion_sampler":"entropy_bound"}'), whose quotes/braces the shell would otherwise strip
    # (vLLM then errors "invalid loads value: {enable_thinking:true}"). Plain flags quote to themselves.
    args.extend(shlex.quote(str(x)) for x in vcfg.get('serve_args', []))
    args.extend(['--host', shlex.quote(host), '--port', str(port)])
    if 'api_key' in vcfg:
        args.extend(['--api-key', shlex.quote(str(vcfg['api_key']))])

    export_prefix = ' '.join(f"export {k}={shlex.quote(v)};" for k, v in env_exports.items())
    ld_glob = "$(ls -d /usr/local/lib/python*/dist-packages/nvidia/*/lib 2>/dev/null | tr '\\n' ':')"
    ld_fix = 'export LD_LIBRARY_PATH="' + ld_glob + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
    cmd = f"{ld_fix}{export_prefix} nohup {' '.join(args)} > {RESULTS}/serve.log 2>&1 & echo $! > {RESULTS}/serve.pid"
    run(cmd, 'serve_start.log', check=True, timeout=60)
    ok = wait_for_url(f'http://{host}:{port}/v1/models', max_wait, 'serve_start.log')
    return {'ok': ok, 'base_url': f'http://{host}:{port}/v1', 'model_id': model_id, 'backend': 'vllm'}


def install_llama_cpp(scfg: Dict[str, Any]) -> None:
    if not scfg.get('install', True):
        return
    lcfg = scfg.get('llama_cpp', {})
    wheel = lcfg.get('wheel', 'llama-cpp-python[server]==0.3.29')
    index = lcfg.get('wheel_index', 'https://abetlen.github.io/llama-cpp-python/whl/cu124')
    # Prebuilt CUDA wheel (--prefer-binary => never compile on the 2-vCPU VM) + hf_hub for download.
    run(f"{sys.executable} -m pip install -q {shlex.quote(wheel)} "
        f"--extra-index-url {shlex.quote(index)} --prefer-binary",
        'install.log', check=True, timeout=int(lcfg.get('install_timeout_seconds', 1800)))
    run(f"{sys.executable} -m pip install -q -U huggingface_hub", 'install.log', check=False, timeout=600)


def start_llama_cpp(config: Dict[str, Any], scfg: Dict[str, Any]) -> Dict[str, Any]:
    model = config['model']
    model_id = model['id']
    lcfg = scfg.get('llama_cpp', {})
    repo = model.get('gguf_repo') or lcfg.get('gguf_repo')
    gguf_file = model.get('gguf_file') or lcfg.get('gguf_file')
    if not repo or not gguf_file:
        raise RuntimeError('llama_cpp backend requires model.gguf_repo and model.gguf_file')
    host, port = scfg['host'], scfg['port']
    max_wait = scfg['startup_timeout_seconds']
    server_args = [str(a) for a in lcfg.get('server_args', ['--n_gpu_layers', '99', '--n_ctx', '4096'])]

    # Download the GGUF. Build the python one-liner with %r for safe Python literals, then
    # shlex.quote the WHOLE program so the shell sees one safe token (robust to quotes in names).
    prog = ('from huggingface_hub import hf_hub_download as d; '
            'print(d(%r, %r, local_dir=%r))' % (repo, gguf_file, str(GGUF_DIR)))
    dl = f"{sys.executable} -c {shlex.quote(prog)}"
    run(dl, 'llama_download.log', check=True, timeout=int(lcfg.get('download_timeout_seconds', 1800)))
    gguf_path = str(GGUF_DIR / gguf_file)

    args = [sys.executable, '-m', 'llama_cpp.server', '--model', shlex.quote(gguf_path),
            '--model_alias', shlex.quote(model_id)] + server_args + \
           ['--host', shlex.quote(host), '--port', str(port)]
    cmd = f"nohup {' '.join(args)} > {RESULTS}/serve.log 2>&1 & echo $! > {RESULTS}/serve.pid"
    run(cmd, 'serve_start.log', check=True, timeout=60)
    ok = wait_for_url(f'http://{host}:{port}/v1/models', max_wait, 'serve_start.log')
    return {'ok': ok, 'base_url': f'http://{host}:{port}/v1', 'model_id': model_id, 'backend': 'llama_cpp'}


# OpenAI-compatible shim over google.colab.ai (free in-Colab Gemini/Gemma — NO GPU, NO download,
# NO API fee). Runs INSIDE the Colab VM (where `from google.colab import ai` exists) and lets
# OpenClaw point at it on loopback exactly like a real model server. NOTE: inference runs on
# Google's backend, so prompts leave the sandbox — this is the fee-free-but-not-contained path.
COLAB_AI_SHIM = r'''
import os, json, time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from google.colab import ai

MODEL = os.environ.get("COLAB_AI_MODEL", "google/gemini-3.5-flash")
app = FastAPI()

@app.get("/v1/models")
def _models():
    try:
        ids = list(ai.list_models())
    except Exception:
        ids = [MODEL]
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "google"} for m in ids]}

def _content(m):
    c = m.get("content", "")
    return c if isinstance(c, str) else json.dumps(c)

@app.post("/v1/chat/completions")
async def _chat(req: Request):
    body = await req.json()
    msgs = body.get("messages", [])
    prompt = "\n\n".join(f"{m.get('role','user')}: {_content(m)}" for m in msgs) or "Hello"
    try:
        text = ai.generate_text(prompt, model_name=MODEL)   # text-to-text; non-streaming
    except Exception as e:
        text = "[colab_ai error] " + repr(e)
    created, model = int(time.time()), (body.get("model") or MODEL)
    # OpenClaw's infer requests stream:true -> answer with SSE (full delta + finish + [DONE]),
    # else it reports "Stream ended without finish_reason" and drops the text.
    if body.get("stream"):
        def _sse():
            head = {"id": "chatcmpl-colabai", "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [{"index": 0,
                    "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
            tail = {"id": "chatcmpl-colabai", "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield "data: " + json.dumps(head) + "\n\n"
            yield "data: " + json.dumps(tail) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_sse(), media_type="text/event-stream")
    return {"id": "chatcmpl-colabai", "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {}}

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("COLAB_AI_HOST", "127.0.0.1"),
                port=int(os.environ.get("COLAB_AI_PORT", "8000")))
'''


def install_colab_ai(scfg: Dict[str, Any]) -> None:
    if not scfg.get('install', True):
        return
    cai = scfg.get('colab_ai', {})
    run(f"{sys.executable} -m pip install -q fastapi uvicorn", 'install.log',
        check=True, timeout=int(cai.get('install_timeout_seconds', 600)))


def start_colab_ai(config: Dict[str, Any], scfg: Dict[str, Any]) -> Dict[str, Any]:
    model_id = config['model']['id']
    cai = scfg.get('colab_ai', {})
    ai_model = cai.get('model', 'google/gemini-2.5-flash')
    host, port = scfg['host'], scfg['port']
    (BASE / 'colab_ai_shim.py').write_text(COLAB_AI_SHIM, encoding='utf-8')
    env = {'COLAB_AI_MODEL': ai_model, 'COLAB_AI_HOST': host, 'COLAB_AI_PORT': str(port)}
    cmd = (f"export COLAB_AI_MODEL={shlex.quote(ai_model)}; export COLAB_AI_HOST={shlex.quote(host)}; "
           f"export COLAB_AI_PORT={port}; nohup {sys.executable} {BASE / 'colab_ai_shim.py'} "
           f"> {RESULTS}/serve.log 2>&1 & echo $! > {RESULTS}/serve.pid")
    run(cmd, 'serve_start.log', check=True, timeout=60, env=env)
    ok = wait_for_url(f'http://{host}:{port}/v1/models', scfg['startup_timeout_seconds'], 'serve_start.log')
    return {'ok': ok, 'base_url': f'http://{host}:{port}/v1', 'model_id': model_id,
            'backend': 'colab_ai', 'ai_model': ai_model}


def install_ollama(scfg: Dict[str, Any]) -> None:
    if not scfg.get('install', True):
        return
    ocfg = scfg.get('ollama', {})
    # Ollama's installer extracts a zstd-compressed tarball; the Colab base image lacks `zstd`
    # ("This version requires zstd for extraction"). apt here must (a) refresh a stale cache and
    # (b) WAIT for the dpkg lock — the OpenClaw installer runs apt concurrently in the background, so
    # DPkg::Lock::Timeout avoids an immediate "could not get lock" failure. Best-effort; `which zstd`
    # logs the outcome.
    run('(apt-get update -qq -o DPkg::Lock::Timeout=300 2>&1 || true); '
        'DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --fix-missing '
        '-o DPkg::Lock::Timeout=300 zstd 2>&1; echo "zstd -> $(command -v zstd || echo MISSING)"',
        'install.log', check=False, timeout=600)
    # Official install script ships a PREBUILT CUDA runner (no compile) and tracks a current
    # llama.cpp — so it serves LFM2.5 tool calls that llama-cpp-python's server can't (it returns
    # structured OpenAI tool_calls via Ollama's own template parser). Idempotent.
    run('curl -fsSL https://ollama.com/install.sh | sh', 'install.log',
        check=True, timeout=int(ocfg.get('install_timeout_seconds', 900)))


def start_ollama(config: Dict[str, Any], scfg: Dict[str, Any]) -> Dict[str, Any]:
    model_id = config['model']['id']            # an Ollama tag, e.g. 'lfm2.5:8b'
    ocfg = scfg.get('ollama', {})
    host, port = scfg['host'], scfg['port']
    max_wait = scfg['startup_timeout_seconds']
    num_ctx = int(ocfg.get('num_ctx', 32768))
    num_parallel = int(ocfg.get('num_parallel', 0))    # 0 = Ollama default (~1 request/model); >0 serves
                                                        # that many requests CONCURRENTLY so a multi-agent
                                                        # fan-out / coordinator-tier depth run executes in
                                                        # PARALLEL (unlike vLLM --max-num-seqs 1 which
                                                        # serializes and times out for block-diffusion).
    # Bind the daemon to the project's loopback :8000 (NEVER :11434 default if it clashes), and set
    # the default context (Ollama caps num_ctx small regardless of the model's max). OLLAMA_HOST is
    # read by BOTH the daemon and the `ollama pull` client.
    hostport = f'{host}:{port}'
    env = {'OLLAMA_HOST': hostport, 'OLLAMA_CONTEXT_LENGTH': str(num_ctx)}
    par_export = ''
    if num_parallel > 0:
        env['OLLAMA_NUM_PARALLEL'] = str(num_parallel)
        par_export = f"export OLLAMA_NUM_PARALLEL={num_parallel}; "
    serve_cmd = (f"export OLLAMA_HOST={shlex.quote(hostport)}; export OLLAMA_CONTEXT_LENGTH={num_ctx}; "
                 + par_export +
                 f"nohup ollama serve > {RESULTS}/serve.log 2>&1 & echo $! > {RESULTS}/serve.pid")
    run(serve_cmd, 'serve_start.log', check=True, timeout=60, env=env)
    # Wait for the daemon, pull the model (blocks until downloaded), then confirm the OpenAI endpoint.
    wait_for_url(f'http://{host}:{port}/api/version', 120, 'serve_start.log')
    run(f"export OLLAMA_HOST={shlex.quote(hostport)}; ollama pull {shlex.quote(model_id)}",
        'ollama_pull.log', check=True, timeout=int(ocfg.get('download_timeout_seconds', 1800)), env=env)
    ok = wait_for_url(f'http://{host}:{port}/v1/models', max_wait, 'serve_start.log')
    return {'ok': ok, 'base_url': f'http://{host}:{port}/v1', 'model_id': model_id, 'backend': 'ollama'}


def start_backend(config: Dict[str, Any]) -> Dict[str, Any]:
    scfg = serve_cfg(config)
    backend = scfg['backend']
    if backend == 'colab_ai':
        install_colab_ai(scfg)
        return start_colab_ai(config, scfg)
    if backend == 'llama_cpp':
        install_llama_cpp(scfg)
        return start_llama_cpp(config, scfg)
    if backend == 'vllm':
        install_vllm(scfg)
        return start_vllm(config, scfg)
    if backend == 'ollama':
        install_ollama(scfg)
        return start_ollama(config, scfg)
    raise ValueError(f'Unknown serve backend: {backend}')


def install_openclaw_bg():
    """Kick the OpenClaw npm installer off in the background; returns (proc, logfile)."""
    oc_cmd = 'curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard'
    oc_log = (RESULTS / 'openclaw_install.log').open('a', encoding='utf-8')
    oc_log.write(f"\n[{now()}] $ {oc_cmd}\n")
    oc_log.flush()
    proc = subprocess.Popen(oc_cmd, shell=True, executable='/bin/bash', stdout=oc_log, stderr=subprocess.STDOUT)
    return proc, oc_log


DEEP_RESEARCH_SKILL = """---
name: deep-research
description: Conduct rigorous, autonomous multi-step research on a topic — build on prior steps in the session, give concrete evidence, and synthesize a final answer. Use for deep-research tasks.
---

# Deep research

You are running a multi-step research task. The steps share one session, so you RETAIN context.

1. Each step builds on the earlier steps in THIS session. When a step says "the above" or
   "synthesize", refer back to your own earlier answers in this conversation — never ask the user
   to re-supply them.
2. Be concrete: give numbers, ranges, and real examples where you can. State uncertainty explicitly
   instead of inventing specifics.
3. Stay strictly on the topic of the request. Do not drift into unrelated domains or generic filler.
4. Keep each answer focused and well-structured (short headers, tight bullets). No padding.
5. Keep context lean: NEVER paste full web-page text into an answer — extract the few facts that
   matter plus the source URL.
6. EXTERNALIZE MEMORY (required for shared-session multi-step research — this keeps the transcript
   bounded no matter how many steps run). After EACH step, you MUST `write` the distilled finding
   (the key fact + source URL) to a markdown note file at `memory/<short-slug>.md` (for example
   `memory/python-version.md`). The `memory/` folder AND the `.md` extension are REQUIRED — only
   `memory/*.md` notes are indexed for search; a file without `.md` is invisible to recall. In a
   LATER step, do NOT restate or re-paste an earlier finding inline and do NOT scroll back through
   the whole transcript — instead RECALL it with `memory_search` (search by topic) and `memory_get`
   (fetch the matching note). Treat your own prose as transient; treat memory as the store of record.
7. For a final synthesis step, RECALL every earlier finding from memory via `memory_search`/
   `memory_get` (not by re-reading the transcript or re-running searches) and integrate them into a
   concise executive summary with a clear, actionable recommendation.
"""


def _workspace_dir() -> Path:
    return Path(os.path.expanduser(os.environ.get('OPENCLAW_WORKSPACE_DIR', '~/.openclaw/workspace')))


# Lean AGENTS.md for small-context models: keeps the essential web-tool directive but drops the
# ~8KB default (which alone overflowed an 8k window). OpenClaw's base system prompt still carries
# the tool/protocol instructions; this only trims the workspace persona/project layer.
LEAN_AGENTS_MD = """# AGENTS

You are an autonomous research assistant. Be concise, concrete, and cite sources.

- You have `web_search` and `web_fetch` tools. For anything about current events, versions,
  prices, or facts you are not certain of, CALL `web_search` FIRST, then cite the source URL(s).
- Never claim you lack internet access when a web tool is available — use it.
- Build on earlier turns in this session; do not ask the user to repeat themselves.
- Keep context lean: do NOT paste full fetched-page text into your reply — extract only the few
  facts that matter plus the source URL.
- EXTERNALIZE MEMORY: in shared-session multi-step research you MUST `write` each distilled finding
  (key fact + source URL) to a markdown note `memory/<short-slug>.md` (the `memory/` folder and the
  `.md` extension are REQUIRED — only `memory/*.md` is indexed for search), and RECALL earlier
  findings with `memory_search` / `memory_get` instead of restating them inline. Never re-paste or
  re-derive a fact you already saved — look it up. This keeps the transcript bounded across steps.
"""


def _lean_workspace() -> None:
    """Shrink the workspace bootstrap so it fits a small context: replace the verbose default
    AGENTS.md with a lean one and BLANK the optional persona files (OpenClaw skips blank files)."""
    try:
        ws = _workspace_dir()
        ws.mkdir(parents=True, exist_ok=True)
        (ws / 'AGENTS.md').write_text(LEAN_AGENTS_MD, encoding='utf-8')
        for name in ('SOUL.md', 'HEARTBEAT.md', 'IDENTITY.md'):
            (ws / name).write_text('', encoding='utf-8')
    except Exception as exc:
        append(RESULTS / 'error.log', f"[{now()}] lean_workspace {exc!r}\n")


def _configure_web_and_identity(ocfg: Dict[str, Any], env: Dict[str, str]):
    """Optionally enable OpenClaw web tools (Brave) + seed the workspace USER.md identity.

    Gated by config so smoke/legacy runs are untouched:
      openclaw.web      -> {enabled, provider=brave, plugin_package, max_results, profile,
                            code_mode (bool), lean_workspace (bool)}
      openclaw.identity -> {name, email?, notes?}

    Two routes to let the model reach web_search (pick via compat.supportsTools in the config):
      - native tools : compat.supportsTools=true,  code_mode=false (tool-trained models, smaller prompt)
      - codeMode     : compat.supportsTools=false, code_mode=true  (no native function-calling)
    The brave provider is an EXTERNAL plugin in current OpenClaw, so it is INSTALLED before being
    enabled/selected (else `tools.web.search.provider brave` -> "provider not available"). All
    best-effort (check=False); openclaw_web.log captures `config get`/`plugins list`/`doctor` so the
    verify run sees what actually stored. BRAVE_API_KEY must already be in this process env (oc_env
    forwards it); only its PRESENCE is logged, never the value.
    """
    web = ocfg.get('web') or {}
    web_applied: List[str] = []
    if web.get('enabled'):
        provider = web.get('provider', 'brave')
        default_pkg = '@openclaw/brave-plugin' if provider == 'brave' else None
        pkg = web.get('plugin_package', default_pkg)
        if pkg:
            run(PATH_PREFIX + f"openclaw plugins install {shlex.quote(pkg)}",
                'openclaw_web.log', check=False, env=env,
                timeout=int(web.get('plugin_install_timeout', 300)))
            web_applied.append(f'plugins.install:{pkg}')
        sets = [
            (f'plugins.entries.{provider}.enabled', 'true'),
            # Trust the freshly-installed external plugin so it loads explicitly (else OpenClaw warns
            # "plugins.allow is empty; discovered non-bundled plugins may auto-load").
            ('plugins.allow', json.dumps([provider])),
            ('tools.web.search.provider', provider),
            ('tools.web.search.enabled', 'true'),
            ('tools.web.fetch.enabled', 'true'),
            ('tools.web.search.maxResults', str(int(web.get('max_results', 5)))),
            ('tools.web.search.timeoutSeconds', '30'),
            # 'coding' profile already includes group:web AND group:runtime (codeMode needs exec).
            ('tools.profile', web.get('profile', 'coding')),
        ]
        if web.get('code_mode'):
            sets.append(('tools.codeMode.enabled', 'true'))
        for key, val in sets:
            run(PATH_PREFIX + f"openclaw config set {shlex.quote(key)} {shlex.quote(val)}",
                'openclaw_web.log', check=False, env=env, timeout=60)
            web_applied.append(key)
        for key in (f'plugins.entries.{provider}.enabled', 'tools.web.search.provider',
                    'tools.web.search.enabled', 'tools.web.fetch.enabled',
                    'tools.web.search.maxResults', 'tools.profile', 'tools.codeMode'):
            run(PATH_PREFIX + f"openclaw config get {shlex.quote(key)}",
                'openclaw_web.log', check=False, env=env, timeout=60)
        run(PATH_PREFIX + 'openclaw plugins list --json || openclaw plugins list || true',
            'openclaw_web.log', check=False, env=env, timeout=120)
        run(PATH_PREFIX + 'openclaw doctor --json || openclaw doctor || true',
            'openclaw_web.log', check=False, env=env, timeout=120)
        append(RESULTS / 'openclaw_web.log',
               f"[{now()}] BRAVE_API_KEY_present_in_env={bool(os.environ.get('BRAVE_API_KEY'))}\n")
        if web.get('lean_workspace'):
            _lean_workspace()

    identity = ocfg.get('identity') or {}
    seeded = False
    if identity.get('name'):
        try:
            ws = _workspace_dir()
            ws.mkdir(parents=True, exist_ok=True)
            name = str(identity['name'])
            parts = [f"# USER\n\n- The user's name is {name}. Always address them as {name}.\n"]
            if identity.get('email'):
                parts.append(f"- Email: {identity['email']}\n")
            if identity.get('notes'):
                parts.append(f"- {identity['notes']}\n")
            (ws / 'USER.md').write_text(''.join(parts), encoding='utf-8')
            seeded = True
        except Exception as exc:
            append(RESULTS / 'error.log', f"[{now()}] USER.md seed {exc!r}\n")
    return web_applied, seeded


def _configure_context(ocfg: Dict[str, Any], env: Dict[str, str]):
    """Apply OpenClaw's BOUNDED-CONTEXT settings for long-horizon tool-heavy research (gated by
    openclaw.context so smoke/legacy runs are untouched). OpenClaw's documented best practice — NOT
    'raise the window':
      - contextPruning is OFF by default for non-Anthropic backends (our vLLM/Ollama/llama.cpp) —
        turning it ON trims/drops OLD tool results (3-5k-tok web pages) from the in-memory prompt.
      - the 'contextWindow/2' budget is reserveTokensFloor eaten from a small window → LOWER it.
      - midTurnPrecheck compacts AFTER a tool result, BEFORE the next call (pre-empts 'Already compacted').
      - cap per-tool-result size; point memory_search at a no-key provider for the loopback Colab.
    All best-effort (check=False); openclaw_context.log captures `config get` so the verify sees what stored.
    """
    ctx = ocfg.get('context') or {}
    applied: List[str] = []
    if not ctx.get('enabled'):
        return applied
    sets = []
    pr = ctx.get('pruning') or {}
    if pr.get('enabled', True):
        sets += [
            ('agents.defaults.contextPruning.mode', str(pr.get('mode', 'cache-ttl'))),
            ('agents.defaults.contextPruning.ttl', str(pr.get('ttl', '5m'))),
            ('agents.defaults.contextPruning.minPrunableToolChars', str(int(pr.get('min_prunable_tool_chars', 8000)))),
        ]
    if 'reserve_tokens_floor' in ctx:
        sets.append(('agents.defaults.compaction.reserveTokensFloor', str(int(ctx['reserve_tokens_floor']))))
    if 'reserve_tokens' in ctx:
        sets.append(('agents.defaults.compaction.reserveTokens', str(int(ctx['reserve_tokens']))))
    sets += [
        ('agents.defaults.compaction.mode', str(ctx.get('compaction_mode', 'safeguard'))),
        ('agents.defaults.compaction.midTurnPrecheck.enabled', 'true' if ctx.get('mid_turn_precheck', True) else 'false'),
    ]
    if 'tool_result_max_chars' in ctx:
        sets.append(('agents.defaults.contextLimits.toolResultMaxChars', str(int(ctx['tool_result_max_chars']))))
    mem_provider = ctx.get('memory_search_provider')
    if mem_provider:
        # Layer 2 (memory externalization): set the search backend. 'none' = OpenClaw's no-egress
        # in-process BM25 index (no key, no model download); 'local' would pull a ~0.6GB embedding
        # model into the ephemeral Colab VM, so we keep BM25 by default.
        # provider sets the SEARCH backend; enabled:true turns memory search on. CRITICAL: memory_search
        # only finds notes saved as MEMORY.md / memory/*.md (markdown -> per-agent SQLite FTS index,
        # 1.5s debounced reindex), so the skill + task instruct the agent to save findings exactly
        # there with a .md extension. (The earlier tools.memory.enabled / agents.defaults.memory.enabled
        # keys were INVALID — config validation rejected them — and unnecessary; the memory tools are on
        # by default. Run C 2026-06-22: the agent wrote memory/<name> with NO .md, so nothing indexed
        # and recall returned empty.)
        sets += [
            ('agents.defaults.memorySearch.provider', str(mem_provider)),
            ('agents.defaults.memorySearch.enabled', 'true'),
        ]
    for key, val in sets:
        run(PATH_PREFIX + f"openclaw config set {shlex.quote(key)} {shlex.quote(val)}",
            'openclaw_context.log', check=False, env=env, timeout=60)
        applied.append(key)
    for key in ('agents.defaults.contextPruning', 'agents.defaults.compaction.reserveTokensFloor',
                'agents.defaults.compaction.reserveTokens', 'agents.defaults.compaction.midTurnPrecheck.enabled',
                'agents.defaults.contextLimits.toolResultMaxChars', 'agents.defaults.memorySearch.provider',
                'agents.defaults.memorySearch.enabled'):
        run(PATH_PREFIX + f"openclaw config get {shlex.quote(key)}",
            'openclaw_context.log', check=False, env=env, timeout=60)
    return applied


def _configure_fanout(ocfg: Dict[str, Any], env: Dict[str, str]) -> List[str]:
    """Apply OpenClaw's multi-level subagent spawn-depth knob (gated by openclaw.fanout, so
    smoke/legacy and single-level fan-out runs are untouched).

    By default OpenClaw caps spawn depth at 1 (the LEAD spawns leaf workers, which CANNOT spawn
    their own children). Raising it to 2 enables the orchestrator pattern: LEAD -> COORDINATOR
    sub-agents -> leaf workers, for research trees too big for one lead to enumerate.

    NOTE: the DOCUMENTED config key is `agents.defaults.subagents.maxSpawnDepth` (verified against
    OpenClaw docs https://docs.openclaw.ai/tools/subagents — NOT the un-namespaced
    `agents.defaults.maxSpawnDepth`). Best-effort (check=False); openclaw_fanout.log captures the
    `config get` round-trip so the verify run sees what actually stored. Default n=1 is a no-op
    (it's already OpenClaw's default), so we only write when max_spawn_depth > 1.
    """
    fan = ocfg.get('fanout') or {}
    applied: List[str] = []
    if not bool(fan.get('multilevel', False)):
        # Multi-level fan-out DISABLED by default (it never completes on the available models — see
        # _task_run + docs/validation_findings.md). Leave maxSpawnDepth at OpenClaw's default 1 so
        # children CANNOT spawn their own children: flat single-level fan-out only (verified working).
        return applied
    max_depth = int(fan.get('max_spawn_depth', 1))
    if max_depth > 1:
        # Also CAP fan-out width: DiffusionGemma over-spawns (a coordinator fired ~12 leaf spawns for
        # 3 sub-questions), and 2 coordinators x ~12 leaves >> the maxConcurrent lane (default 8) →
        # the whole tree queues and STALLS before synthesis (verified 2026-06-22). maxChildrenPerAgent
        # (docs default 5, range 1-20) bounds ACTIVE children per agent so a runaway coordinator can't
        # flood the lane; maxConcurrent (docs default 8) is raised to give the well-formed tree
        # (G coordinators x ~3 leaves) headroom. All documented agents.defaults.subagents keys.
        sets = [
            ('agents.defaults.subagents.maxSpawnDepth', str(max_depth)),
            ('agents.defaults.subagents.maxChildrenPerAgent', str(int(fan.get('max_children_per_agent', 4)))),
            ('agents.defaults.subagents.maxConcurrent', str(int(fan.get('max_concurrent', 12)))),
        ]
        for key, val in sets:
            run(PATH_PREFIX + f"openclaw config set {shlex.quote(key)} {shlex.quote(val)}",
                'openclaw_fanout.log', check=False, env=env, timeout=60)
            run(PATH_PREFIX + f"openclaw config get {shlex.quote(key)}",
                'openclaw_fanout.log', check=False, env=env, timeout=60)
            applied.append(key)
    return applied


def configure_openclaw(config: Dict[str, Any], serve_state: Dict[str, Any]) -> Dict[str, Any]:
    ocfg = config.get('openclaw', {})
    model_id = config['model']['id']
    provider_id = ocfg.get('provider_id', 'vllm')
    model_ref = ocfg.get('model_ref') or f'{provider_id}/{model_id}'
    gateway_port = int(ocfg.get('gateway_port', 18789))
    env = oc_env(config)
    os.environ.update(env)  # so the gateway started later in THIS process inherits the token

    onboard = (
        PATH_PREFIX
        + 'openclaw onboard --non-interactive --accept-risk --mode local '
        + '--auth-choice custom-api-key '
        + f'--custom-provider-id {shlex.quote(provider_id)} '
        + f'--custom-base-url {shlex.quote(serve_state["base_url"])} '
        + f'--custom-model-id {shlex.quote(model_id)} '
        + '--custom-compatibility openai --custom-api-key "${VLLM_API_KEY}" --custom-text-input '
        + f'--gateway-port {gateway_port} --gateway-bind loopback '
        + '--gateway-auth token --gateway-token-ref-env OPENCLAW_GATEWAY_TOKEN '
        # NOTE: skills are NOT skipped — the native agent path (openclaw agent) uses them.
        + '--skip-daemon --skip-channels --skip-health --skip-ui --json'
    )
    run(onboard, 'openclaw_config.log', check=False, env=env, timeout=300)

    # Install a NATIVE OpenClaw skill (best practice) so the agent has a deep-research methodology
    # instead of a hand-rolled Python loop. A SKILL.md under ~/.openclaw/skills/<name>/ is
    # auto-discovered (managed/local root). The agent loads it at session start.
    try:
        skill_dir = Path(os.path.expanduser('~/.openclaw/skills/deep-research'))
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / 'SKILL.md').write_text(DEEP_RESEARCH_SKILL, encoding='utf-8')
        # Scope the agent to ONLY deep-research: the ~20 bundled skills otherwise inject ~8.9k tokens
        # and overflow a small model's prompt. Best-effort (config shape varies by version); the large
        # n_ctx is the safety net if this allowlist key isn't honored. check=False so it never aborts.
        run(PATH_PREFIX + 'openclaw config set agents.defaults.skills \'["deep-research"]\'',
            'openclaw_config.log', check=False, env=env, timeout=60)
        run(PATH_PREFIX + 'openclaw skills list', 'openclaw_skills.log', check=False, env=env, timeout=120)
    except Exception as exc:
        append(RESULTS / 'error.log', f"[{now()}] skill install {exc!r}\n")

    # CRITICAL infer fixes for a local OpenAI-compat backend (validated): without
    # requiresStringContent the gateway returns an empty completion, and without
    # maxTokens < contextWindow the request overflows -> incomplete_result. Only the
    # `models.providers.<id>.models[0]...` index form is valid. Applied only when the config
    # supplies a `compat` block (legacy DiffusionGemma config omits it).
    compat = ocfg.get('compat')
    applied = []
    if compat:
        base = f'models.providers.{provider_id}.models[0]'
        pairs = [
            (f'{base}.compat.requiresStringContent', 'true' if compat.get('requiresStringContent', True) else 'false'),
            (f'{base}.compat.supportsTools', 'true' if compat.get('supportsTools', False) else 'false'),
            (f'{base}.maxTokens', str(int(compat.get('maxTokens', 1024)))),
            (f'{base}.contextWindow', str(int(compat.get('contextWindow', 4096)))),
        ]
        for key, val in pairs:
            run(PATH_PREFIX + f"openclaw config set {shlex.quote(key)} {shlex.quote(val)}",
                'openclaw_config.log', check=False, env=env, timeout=60)
            applied.append(key)

    # Optional deep-research wiring (web search + identity), gated by config (off for smoke/legacy).
    web_applied, identity_seeded = _configure_web_and_identity(ocfg, env)
    # Optional bounded-context settings (pruning/reserve/compaction) for long-horizon research.
    context_applied = _configure_context(ocfg, env)
    # Optional multi-level fan-out: raise the subagent spawn-depth cap so the lead can use COORDINATOR
    # sub-agents (each spawning its own leaf workers) for research trees too big for one lead.
    fanout_applied = _configure_fanout(ocfg, env)

    run(PATH_PREFIX + 'openclaw config file', 'openclaw_config.log', check=False, env=env, timeout=60)
    run(PATH_PREFIX + 'openclaw models list --json', 'openclaw_models.log', check=False, env=env, timeout=120)
    return {'model_ref': model_ref, 'provider_id': provider_id, 'gateway_port': gateway_port,
            'gateway_token_set': bool(env['OPENCLAW_GATEWAY_TOKEN']), 'compat_applied': applied,
            'web_applied': web_applied, 'identity_seeded': identity_seeded, 'context_applied': context_applied,
            'fanout_applied': fanout_applied}


def start_openclaw_gateway(config: Dict[str, Any]) -> Dict[str, Any]:
    ocfg = config.get('openclaw', {})
    gateway_port = int(ocfg.get('gateway_port', 18789))
    env = oc_env(config)
    cmd = PATH_PREFIX + 'setsid nohup openclaw gateway run > ' + str(RESULTS / 'openclaw_gateway.log') + \
          ' 2>&1 < /dev/null & echo $! > ' + str(RESULTS / 'openclaw_gateway.pid')
    run(cmd, 'openclaw_gateway_start.log', check=False, env=env, timeout=60)
    time.sleep(8)
    run(PATH_PREFIX + 'openclaw gateway status --json || openclaw gateway status || true',
        'openclaw_gateway_status.log', check=False, env=env, timeout=120)
    return {'gateway_port': gateway_port, 'pid_file': str(RESULTS / 'openclaw_gateway.pid')}


def _infer_cmd(model_ref: str, prompt_text: str, transport: str) -> str:
    # transport 'gateway' -> --gateway; anything else -> direct infer (no flag), the robust path.
    flag = '--gateway ' if transport == 'gateway' else ''
    return (PATH_PREFIX + 'openclaw infer model run ' + flag
            + '--model ' + shlex.quote(model_ref) + ' --prompt ' + shlex.quote(prompt_text) + ' --json')


def _agent_cmd(model_ref: str, message: str, session_key: str, timeout_s: int) -> str:
    # Native OpenClaw agent turn (best practice, vs. one-shot infer): `--local` runs the EMBEDDED
    # agent (no gateway -> sidesteps the connected-no-operator-scope issue), and a shared
    # `--session-key` makes OpenClaw keep conversation context server-side across steps. Loaded
    # skills (e.g. deep-research) are applied automatically — no hand-rolled chain-of-thought.
    return (PATH_PREFIX + 'openclaw agent --local --agent main '
            + '--session-key ' + shlex.quote(session_key) + ' '
            + '--model ' + shlex.quote(model_ref) + ' '
            + '--message ' + shlex.quote(message) + ' '
            + '--timeout ' + str(int(timeout_s)) + ' --json')


# ---------------------------------------------------------------------------
# Generic detached-phase scaffolding
# ---------------------------------------------------------------------------

def _launch_worker(worker: str, status_path: Path, done_path: Path, log_name: str, token: str) -> None:
    """Fast action: clear markers, launch `--worker <worker>` detached, return immediately."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    for marker in (done_path, status_path):
        if marker.exists():
            marker.unlink()
    write_status(status_path, 'launching')
    worker_log = (RESULTS / log_name).open('a', encoding='utf-8')
    subprocess.Popen([sys.executable, str(SELF_PATH), '--worker', worker],
                     stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True)
    print(token)


def _emit_status(status_path: Path, done_path: Path, token: str, extra_detail=None) -> None:
    """Fast action: print `<TOKEN>=running|ready|failed` for the launcher's poll loop."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    detail: Dict[str, Any] = {}
    if done_path.exists():
        try:
            done = json.loads(done_path.read_text(encoding='utf-8'))
        except Exception:
            done = {'ok': False}
        detail['done'] = done
        state = 'ready' if done.get('ok') else 'failed'
    else:
        if status_path.exists():
            try:
                detail['status'] = json.loads(status_path.read_text(encoding='utf-8'))
            except Exception:
                pass
        state = 'running'
    if extra_detail:
        detail.update(extra_detail)
    print(f'{token}={state}')
    print(f'{token}_DETAIL=' + json.dumps(detail))


# ---------------------------------------------------------------------------
# Phase: bootstrap (serve backend + onboard OpenClaw)
# ---------------------------------------------------------------------------

def bootstrap() -> None:
    _launch_worker('bootstrap', STATUS_PATH, DONE_PATH, 'bootstrap_worker.log', 'BOOTSTRAP_LAUNCHED')


def _bootstrap_run() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_json(CONFIG_PATH)
    scfg = serve_cfg(config)
    manifest: Dict[str, Any] = {'started_at': now(), 'action': 'bootstrap', 'backend': scfg['backend']}
    try:
        write_status(STATUS_PATH, 'environment')
        collect_environment()

        # Overlap the slow installs: OpenClaw installer in the background, serve backend foreground.
        write_status(STATUS_PATH, 'installing')
        oc_proc, oc_log = (None, None)
        if config.get('openclaw', {}).get('install', True):
            oc_proc, oc_log = install_openclaw_bg()

        write_status(STATUS_PATH, 'serving')
        serve_state = start_backend(config)
        manifest['serve'] = serve_state

        if oc_proc is not None:
            oc_proc.wait()
            oc_log.flush(); oc_log.close()
        run(PATH_PREFIX + 'openclaw --version', 'openclaw_install.log', check=False, timeout=60)

        write_status(STATUS_PATH, 'configuring_openclaw')
        manifest['openclaw_config'] = configure_openclaw(config, serve_state)
        manifest['openclaw_gateway'] = start_openclaw_gateway(config)
        manifest['ok'] = bool(serve_state.get('ok'))
    except Exception as exc:
        manifest['ok'] = False
        manifest['error'] = repr(exc)
        append(RESULTS / 'error.log', f"[{now()}] bootstrap {exc!r}\n")
    manifest['finished_at'] = now()
    write_json(RESULTS / 'manifest.json', manifest)
    ok = bool(manifest.get('ok'))
    write_status(STATUS_PATH, 'done', {'ok': ok})
    DONE_PATH.write_text(json.dumps({'ok': ok, 'time': now()}), encoding='utf-8')
    bundle()


def _probe_serve_up(config: Dict[str, Any]) -> bool:
    try:
        scfg = serve_cfg(config)
        http_get_json(f"http://{scfg['host']}:{scfg['port']}/v1/models", timeout_s=3)
        return True
    except Exception:
        return False


def bootstrap_status() -> None:
    try:
        up = {'serve_up': _probe_serve_up(load_json(CONFIG_PATH))}
    except Exception:
        up = {'serve_up': False}
    _emit_status(STATUS_PATH, DONE_PATH, 'BOOTSTRAP_STATE', up)


# ---------------------------------------------------------------------------
# Phase: prompt (single infer — smoke test), detached + polled
# ---------------------------------------------------------------------------

def prompt() -> None:
    _launch_worker('prompt', PROMPT_STATUS_PATH, PROMPT_DONE_PATH, 'prompt_worker.log', 'PROMPT_LAUNCHED')


def _prompt_run() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = load_json(RESULTS / 'manifest.json', default={})
    ok = False
    try:
        config = load_json(CONFIG_PATH)
        task = load_json(TASK_PATH)
        prompt_text = task.get('prompt') or 'Reply with exactly: smoke-ok'
        provider_id = config.get('openclaw', {}).get('provider_id', 'vllm')
        model_ref = config.get('openclaw', {}).get('model_ref') or f"{provider_id}/{config['model']['id']}"
        transport = task.get('transport', 'gateway')
        result = run(_infer_cmd(model_ref, prompt_text, transport), 'openclaw_infer.txt',
                     check=False, env=oc_env(config), timeout=int(task.get('timeout_seconds', 900)))
        raw = (RESULTS / 'openclaw_infer.txt').read_text(encoding='utf-8', errors='replace')
        text = extract_infer_text(raw)
        if text is not None:
            write_json(RESULTS / 'openclaw_infer.json', {'text': text})
        ok = result['returncode'] == 0 and text is not None
        manifest['prompt'] = {'model_ref': model_ref, 'transport': transport,
                              'returncode': result['returncode'], 'got_text': text is not None}
    except Exception as exc:
        manifest['error'] = repr(exc)
        append(RESULTS / 'error.log', f"[{now()}] prompt {exc!r}\n")
    manifest['ok'] = bool(manifest.get('ok', True)) and ok
    manifest['finished_at'] = now()
    write_json(RESULTS / 'manifest.json', manifest)
    write_status(PROMPT_STATUS_PATH, 'done', {'ok': ok})
    PROMPT_DONE_PATH.write_text(json.dumps({'ok': ok, 'time': now()}), encoding='utf-8')
    bundle()


def prompt_status() -> None:
    _emit_status(PROMPT_STATUS_PATH, PROMPT_DONE_PATH, 'PROMPT_STATE')


# ---------------------------------------------------------------------------
# Phase: task (autonomous, time-consuming job — deep research), detached + polled
# ---------------------------------------------------------------------------

def _is_substantive_synthesis(text: str) -> bool:
    """A real lead answer vs an OpenClaw control/empty marker. compaction.memoryFlush emits a silent
    'NO_REPLY' turn (terminal stopReason, no toolCall) BEFORE compaction; that turn must NOT be
    mistaken for the lead's synthesis (verified 2026-06-22: a depth-2 run produced the full table but
    a later NO_REPLY turn was captured instead). Exclude NO_REPLY and trivially-short/empty turns from
    both synthesis selection and the early-exit trigger."""
    t = (text or '').strip()
    return len(t) >= 12 and t.upper().strip('. ') not in {'NO_REPLY', 'NOREPLY', 'NO REPLY'}


def _lead_completed_events(session_key: str) -> List[tuple]:
    """Scan OpenClaw's live trajectory for the LEAD session's `model.completed` events.

    Shared detection used by both the synthesis recovery and the early-exit poll: find every
    trajectory event with sessionKey == 'agent:main:<session_key>' (children are
    'agent:main:subagent:...'), type == 'model.completed', and non-empty data.assistantTexts;
    return them as (seq, text, is_final) tuples (unsorted). Best-effort: any failure returns [].

    `is_final` marks the lead's TERMINAL turn — its synthesis — vs an intermediate narration emitted
    between tool calls: the turn's last assistant message ended the turn (terminal stopReason) AND
    issued no further toolCall. The bare presence of assistantTexts is NOT a "done" signal —
    DiffusionGemma emits things like "I am still waiting for the other sub-agents..." mid-orchestration,
    and an over-eager early-exit on those truncates the run (verified 2026-06-22). The terminal
    stopReason + no-toolCall pattern uniquely identifies the real final table in a known-good run."""
    want = 'agent:main:' + session_key
    final_reasons = {'stop', 'end_turn', 'endturn', 'end', 'completed', 'stop_sequence', 'done'}
    events: List[tuple] = []
    try:
        root = Path(os.path.expanduser('~/.openclaw/agents'))
        if not root.exists():
            return events
        for traj in root.glob('*/sessions/*.trajectory.jsonl'):
            try:
                lines = traj.read_text(encoding='utf-8', errors='replace').splitlines()
            except Exception:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get('sessionKey') != want or ev.get('type') != 'model.completed':
                    continue
                data = ev.get('data') or {}
                texts = data.get('assistantTexts') or []
                joined = "\n".join(t for t in texts if isinstance(t, str) and t.strip()).strip()
                if not joined:
                    continue
                is_final = False
                asst = [m for m in (data.get('messagesSnapshot') or []) if m.get('role') == 'assistant']
                if asst:
                    last = asst[-1]
                    sr = str(last.get('stopReason') or '').lower()
                    has_tool = any(isinstance(b, dict) and b.get('type') == 'toolCall'
                                   for b in (last.get('content') or []))
                    is_final = (sr in final_reasons) and not has_tool and _is_substantive_synthesis(joined)
                events.append((int(ev.get('seq', 0) or 0), joined, is_final))
    except Exception:
        return events
    return events


def _lead_synthesis_from_trajectory(session_key: str) -> Optional[str]:
    """Recover the LEAD agent's final synthesis from OpenClaw's server-side trajectory.

    `openclaw agent --local --json` can hang for many minutes AFTER the lead has already produced
    its answer when subagents are spawned (the CLI does not self-exit until the children are
    reaped), so the subprocess is killed by our timeout and never prints the `--json` payload —
    `extract_agent_text(task_lead.log)` then finds nothing. But the runtime writes the trajectory
    live to disk, so the synthesis survives the kill. Prefer the latest TERMINAL synthesis
    (is_final) so an intermediate "still waiting..." turn never wins; fall back to the latest
    assistantTexts only if no terminal turn was captured. Best-effort: any failure returns None."""
    events = _lead_completed_events(session_key)
    if not events:
        return None
    finals = [e for e in events if e[2]]
    pick = max(finals, key=lambda e: e[0]) if finals else max(events, key=lambda e: e[0])
    return pick[1]


def _fanout_lead_message(steps: List[str]) -> str:
    """Lead-agent prompt for orchestration='subagent-fanout': delegate each sub-question to an
    ISOLATED sub-agent (sessions_spawn/sessions_yield) so raw web pages stay in the child context and
    only a distilled summary returns to the lead — the structural bounded-context fix (Layer 3)."""
    subqs = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    n = len(steps)
    return (
        "You are a research LEAD. Coordinate ISOLATED sub-agents so raw web pages never fill your own "
        "context. Sub-questions to answer:\n\n" + subqs + "\n\n"
        # PARALLEL fan-out: spawn ALL children first (so their web searches overlap), THEN collect.
        f"PHASE 1 — SPAWN ALL ({n} sub-questions): make {n} `sessions_spawn` calls in immediate "
        "succession, ONE per sub-question, BEFORE calling sessions_yield even once. Do NOT spawn a "
        "child, yield it, then spawn the next — that serializes them. Fire off every spawn first so the "
        "workers search the web concurrently. Each `sessions_spawn` call uses context:\"isolated\" and a "
        "`task` that tells the worker to run AT MOST 2 web_search/web_fetch calls to answer that ONE "
        "sub-question and to END its reply with a concise (<=300 token) summary containing the answer "
        "plus the exact source URL(s). The worker must NOT loop or keep searching once it has the answer.\n"
        f"PHASE 2 — COLLECT ALL: only AFTER all {n} children are spawned, call `sessions_yield` "
        f"repeatedly ({n} times) to receive each worker's summary, one yield per spawned child. Do NOT "
        "poll with sessions_list, sessions_history, or sleep — `sessions_yield` is the waiting primitive.\n"
        "PHASE 3 — SYNTHESIZE: after every worker has returned, write a FINAL Markdown answer addressing "
        "every sub-question with its value and source URL (use a table when appropriate). Never paste raw "
        "page text — only the distilled findings and citations."
    )


def _fanout_depth_clause(max_depth: int) -> str:
    """Extra lead instruction for MULTI-LEVEL fan-out (only when the config raised
    agents.defaults.subagents.maxSpawnDepth above 1). Empty for depth<=1 so the single-level
    fan-out prompt is byte-identical to before. When there are many sub-questions, the lead MAY
    insert a COORDINATOR tier: spawn a few isolated coordinators, each owning a slice of the
    sub-questions and itself using sessions_spawn/sessions_yield to delegate to leaf workers — so
    every level stays context-bounded (no level ever holds all the raw pages)."""
    if max_depth <= 1:
        return ""
    return (
        "\n\nOPTIONAL MULTI-LEVEL FAN-OUT (allowed: spawn depth up to " + str(int(max_depth)) + "): "
        "if there are MANY sub-questions, you MAY instead spawn a SMALL number (2-3) of COORDINATOR "
        "sub-agents with context:\"isolated\", giving each coordinator a SLICE of the sub-questions. "
        "Instruct each coordinator to itself use `sessions_spawn` (context:\"isolated\") to delegate "
        "each of its sub-questions to a leaf worker and `sessions_yield` to collect those summaries, "
        "then return ONLY a distilled, cited summary of its whole slice to you. This keeps EVERY "
        "level context-bounded — no single agent ever holds all the raw pages. For a handful of "
        "sub-questions, spawning leaf workers directly (as above) is still fine."
    )


def _fanout_lead_message_multilevel(steps: List[str], max_depth: int) -> str:
    """Lead-agent prompt for MULTI-LEVEL (depth>=2) subagent fan-out.

    Replaces (does NOT append to) the flat lead prompt when the config raised
    agents.defaults.subagents.maxSpawnDepth above 1 AND there are enough sub-questions
    (>=4) to warrant a COORDINATOR tier. The flat prompt let DiffusionGemma spawn leaves
    directly and skip coordinators because that tier was only an OPTIONAL "you MAY instead"
    addendum; the strong base instruction won. Here the LEAD->COORDINATOR->leaf tree is the
    ONLY sanctioned shape and there is no competing flat instruction.

    The sub-questions are pre-partitioned in Python into G balanced contiguous coordinator
    GROUPS, and the EXACT verbatim `task` string for each coordinator is embedded so the model
    only COPIES a known-good string into sessions_spawn rather than inventing structure (the
    single highest-leverage lever against a model that ignores soft guidance). Each coordinator
    task itself drives its leaf spawns -> that is what actually produces the third tier;
    maxSpawnDepth>=2 merely permits it. Context stays bounded: raw pages live only in leaves,
    coordinators return short cited summaries, and the lead sees only the G coordinator summaries
    before emitting one final cited table.
    """
    n = len(steps)
    md = int(max_depth)

    # ~3 sub-questions per coordinator; clamp G to 2..4 and never exceed n.
    # Integer ceil(n/3) without a math import == (n + 2) // 3.
    g = (n + 2) // 3
    if g < 2:
        g = 2
    if g > 4:
        g = 4
    if g > n:
        g = n

    # Contiguous balanced partition: the first (n % g) groups get one extra sub-question.
    base = n // g
    rem = n % g
    groups = []  # type: List[List[int]]
    pos = 0
    for gi in range(g):
        size = base + (1 if gi < rem else 0)
        groups.append([pos + 1 + k for k in range(size)])
        pos += size

    # Full master list for the lead's reference ONLY (clearly labelled so it is not mistaken
    # for the work list and answered inline).
    master = "\n".join("%d. %s" % (i, s) for i, s in enumerate(steps, 1))

    # Plain BEGIN-TASK / END-TASK markers (no angle brackets) so the model does not confuse
    # them with tool-call or XML syntax. The embedded coordinator strings use only single
    # quotes / parentheses and NO double quotes, so they drop into a JSON `task` field cleanly.
    blocks = []
    assign_rows = []
    for gi, idxs in enumerate(groups, 1):
        m = len(idxs)
        sub_lines = "\n".join("  %d. %s" % (j, steps[j - 1]) for j in idxs)
        nums = ", ".join(str(j) for j in idxs)
        assign_rows.append("  - COORDINATOR-%d owns sub-question(s): %s" % (gi, nums))
        coord_task = (
            "You are COORDINATOR-%d of %d, a research sub-manager. You own EXACTLY %d "
            "sub-question(s):\n%s\n"
            "PROTOCOL (follow in order; do NOT deviate):\n"
            "1. Do NOT call web_search or web_fetch yourself, and do NOT answer from memory. "
            "You delegate ONLY.\n"
            "2. SPAWN: make EXACTLY %d sessions_spawn call(s) in immediate succession, ONE per "
            "sub-question above, each with context isolated, BEFORE any sessions_yield. Each "
            "spawned LEAF worker's task must instruct it to run AT MOST 2 web_search/web_fetch "
            "calls to answer that ONE sub-question, then END with a concise (<=200 token) summary "
            "giving the answer plus the exact source URL(s), and to STOP (no looping).\n"
            "3. COLLECT: only AFTER all %d leaf workers are spawned, call sessions_yield EXACTLY "
            "%d time(s) to receive each leaf summary. Do NOT poll, sleep, or use sessions_list/"
            "sessions_history -- sessions_yield is the wait primitive.\n"
            "4. RETURN: reply with ONLY a short cited summary, one line per sub-question giving "
            "its number, its answer, and the source URL. Do NOT paste raw page text and do NOT "
            "spawn further coordinators."
            % (gi, g, m, sub_lines, m, m, m)
        )
        blocks.append(
            "COORDINATOR-%d -- make ONE sessions_spawn call with context isolated, "
            "taskName \"coordinator-%d\", and `task` set to the EXACT string between the "
            "BEGIN-TASK and END-TASK marker lines below (copy it verbatim; the marker lines "
            "themselves are NOT part of the task):\n"
            "BEGIN-TASK\n%s\nEND-TASK"
            % (gi, gi, coord_task)
        )

    assignment = "\n".join(assign_rows)
    coord_blocks = "\n\n".join(blocks)

    return (
        "You are the research LEAD (a delegation MANAGER) running a MANDATORY 3-level fan-out "
        "(spawn depth up to %d): LEAD -> COORDINATOR -> leaf worker. This structure is REQUIRED, "
        "not optional. You DELEGATE; you do not research.\n\n"
        "The full list of %d sub-questions is shown below FOR YOUR REFERENCE ONLY -- do NOT "
        "answer any of them yourself:\n\n%s\n\n"
        "HARD RULES (violating ANY of these is a failure):\n"
        "- Do NOT call web_search or web_fetch yourself -- not even once.\n"
        "- Do NOT answer any sub-question from memory.\n"
        "- Do NOT spawn leaf/worker sub-agents directly. You spawn ONLY coordinators.\n"
        "- You will spawn EXACTLY %d coordinator(s) -- no more, no fewer.\n"
        "- Do NOT rewrite, merge, or re-partition the sub-questions, and do NOT edit the "
        "coordinator task strings -- paste each one VERBATIM.\n\n"
        "This FIXED partition assigns each sub-question to exactly one coordinator (do not "
        "change it):\n%s\n\n"
        "PHASE 1 -- SPAWN EXACTLY %d COORDINATOR(S): make %d sessions_spawn call(s) in immediate "
        "succession (BEFORE any sessions_yield), one per block below, each using context isolated "
        "and the EXACT pre-written task string copied verbatim into the `task` field. Fire all %d "
        "spawns first so the coordinators work concurrently. The blocks are:\n\n%s\n\n"
        "PHASE 2 -- COLLECT: only AFTER all %d coordinators are spawned, call sessions_yield "
        "EXACTLY %d time(s) -- one per coordinator -- to receive each coordinator's short cited "
        "summary. sessions_yield BLOCKS until a child finishes; never poll or sleep.\n"
        "PHASE 3 -- SYNTHESIZE: after all %d coordinators have returned, write a FINAL cited "
        "Markdown TABLE with a row for EVERY one of the %d sub-questions: columns = sub-question, "
        "answer/value, source URL. Use ONLY the coordinators' summaries -- never paste raw page "
        "text and never run any search yourself."
        % (md, n, master, g, assignment, g, g, g, coord_blocks, g, g, g, n)
    )


def _trajectory_fingerprint() -> tuple:
    """(latest mtime, total size) across all lead+child trajectory files — a cheap "has anything
    been written lately?" probe for the early-exit silence check. A stable fingerprint over a
    sustained window means no child is still searching and no further announce-driven lead cycle is
    coming (the CLI is just hanging while children are reaped)."""
    root = Path(os.path.expanduser('~/.openclaw/agents'))
    latest = 0.0
    total = 0
    try:
        for t in root.glob('*/sessions/*.trajectory.jsonl'):
            st = t.stat()
            latest = max(latest, st.st_mtime)
            total += st.st_size
    except Exception:
        pass
    return (round(latest, 2), total)


def _run_lead_with_early_exit(cmd: str, session_key: str, hard_timeout: int, log_name: str,
                              env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run the fan-out LEAD CLI but stop as soon as it has finished its synthesis.

    `openclaw agent --local --json` keeps the process alive for many minutes AFTER the lead has
    already produced its final answer once subagents are spawned (it doesn't self-exit until the
    children are reaped), so blocking on it to `hard_timeout` wastes up to ~8 min/run. Instead we
    launch it detached in its own process group and POLL the live trajectory for the lead's
    `model.completed` synthesis (the SAME detection `_lead_synthesis_from_trajectory` uses), then
    kill the group once the lead is done.

    Done heuristic: the lead runs in SEVERAL announce-driven cycles as each child returns, so a
    single terminal turn is NOT enough — an early "still waiting for the other sub-agents..." cycle
    is itself terminal (stopReason stop, no toolCall). Stop only once BOTH hold: (a) a TERMINAL lead
    synthesis exists (model.completed, terminal stopReason, no further toolCall — see
    _lead_completed_events), AND (b) the WHOLE trajectory has been SILENT for `silence_s` (no child
    still searching, no further lead cycle — the CLI is merely hanging while children are reaped).
    `hard_timeout` is the fallback cap. Returns {"returncode": 0} on early/clean/natural exit,
    {"returncode": 124} if the cap elapses with no terminal synthesis — shaped like `run()`'s result
    so the caller can keep reading r["returncode"]."""
    poll_s = 5
    silence_s = int(os.environ.get('OCDG_EARLYEXIT_SILENCE_S', '180'))  # sustained trajectory quiet
                     # that means no child is still working and no further announce-driven lead cycle is
                     # coming. Must EXCEED the slowest single agent turn: a MULTI-LEVEL tree (lead ->
                     # coordinator -> leaf) on slow DiffusionGemma had >75s gaps mid-orchestration, so a
                     # short window killed it on an intermediate "waiting for coordinator-2..." turn
                     # (verified 2026-06-23). 180s default; override via OCDG_EARLYEXIT_SILENCE_S.
    log_path = RESULTS / log_name
    append(log_path, f"\n[{now()}] $ {cmd}\n")
    # Same env merge as run(): start from the process env, overlay the caller's oc_env() overrides.
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.Popen(cmd, shell=True, executable='/bin/bash',
                            stdout=log_path.open('a', encoding='utf-8'),
                            stderr=subprocess.STDOUT, start_new_session=True, env=merged_env)
    deadline = time.time() + hard_timeout
    last_activity = time.time()
    prev_fp: Optional[tuple] = None
    have_synthesis = False
    try:
        while True:
            # Natural exit (lead self-exited, or crashed): honor whatever it did.
            if proc.poll() is not None:
                return {'returncode': 0 if proc.returncode == 0 else proc.returncode}
            if time.time() >= deadline:
                break  # hard cap -> fall through to kill + rc=124
            fp = _trajectory_fingerprint()
            if fp != prev_fp:            # a child or lead cycle just wrote -> still active
                prev_fp = fp
                last_activity = time.time()
            # Need a TERMINAL synthesis AND sustained silence (no more announce-driven cycles coming).
            if any(e[2] for e in _lead_completed_events(session_key)):
                have_synthesis = True
                if time.time() - last_activity >= silence_s:
                    append(log_path, f"[{now()}] early-exit: terminal synthesis + {silence_s}s trajectory silence; stopping\n")
                    break
            time.sleep(poll_s)
    except Exception as exc:
        append(log_path, f"[{now()}] early-exit poll error: {exc!r}\n")
    finally:
        _kill_process_group(proc)
    timed_out = time.time() >= deadline and not have_synthesis
    return {'returncode': 124 if timed_out else 0}


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Best-effort SIGTERM the detached lead's whole process group, then SIGKILL after a grace.

    The lead spawns subagent children in the same group (start_new_session=True), so killing the
    group reaps them too instead of leaving orphaned CLIs holding the model busy."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)  # short grace for graceful shutdown
        return
    except Exception:
        pass
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def task() -> None:
    _launch_worker('task', TASK_STATUS_PATH, TASK_DONE_PATH, 'task_worker.log', 'TASK_LAUNCHED')


def _task_run() -> None:
    """Heavy autonomous task (deep research), executed detached on the VM.

    NATIVE OpenClaw agent path (best practice — not a hand-rolled chain-of-thought): each step is one
    `openclaw agent --local` turn sharing a single --session-key, so OpenClaw keeps conversation
    context server-side across steps (a later "synthesize the above" step actually has the above).
    The agent applies loaded skills (e.g. the deep-research skill installed in configure_openclaw).
    Answers accumulate into research_result.md.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = load_json(RESULTS / 'manifest.json', default={})
    try:
        write_status(TASK_STATUS_PATH, 'running')
        config = load_json(CONFIG_PATH)
        task_cfg = load_json(TASK_PATH)
        provider_id = config.get('openclaw', {}).get('provider_id', 'vllm')
        model_ref = config.get('openclaw', {}).get('model_ref') or f"{provider_id}/{config['model']['id']}"
        steps = task_cfg.get('steps') or [task_cfg.get('prompt') or task_cfg.get('topic') or 'Summarize your capabilities.']
        total_budget = int(task_cfg.get('timeout_seconds', 1800))
        per_step = int(task_cfg.get('step_timeout_seconds', max(180, total_budget // max(1, len(steps)))))
        # One shared session so the agent retains context across steps (server-side, not in Python).
        session_key = task_cfg.get('session_key') or f"research-{int(time.time())}"
        env = oc_env(config)

        orchestration = task_cfg.get('orchestration', 'shared-session')
        out_md = RESULTS / 'research_result.md'
        out_md.write_text(f"# Autonomous research result\n\n- Topic: {task_cfg.get('topic', '')}\n"
                          f"- Model: {model_ref}\n- Engine: openclaw agent (--local, session "
                          f"{session_key}, orchestration={orchestration})\n- Started: {now()}\n",
                          encoding='utf-8')
        step_results = []
        if orchestration == 'subagent-fanout':
            # Layer 3: one LEAD turn fans each sub-question out to an isolated sub-agent so raw pages
            # stay in the child context; only distilled summaries return. Single long turn (no Python
            # poll loop — the lead uses sessions_yield), so give it the whole budget.
            lead_to = int(task_cfg.get('lead_timeout_seconds', total_budget))
            # Multi-level (LEAD -> COORDINATOR -> leaf) fan-out is DISABLED by default. It is gated on
            # an explicit opt-in flag openclaw.fanout.multilevel (default False) because live
            # verification (2026-06-23) proved it NEVER COMPLETES on the available models: the tree
            # always FORMS, but DiffusionGemma-26B is too slow (one coordinator always lags past budget),
            # LFM2.5-8B is too weak (echoes the spawn args as text instead of spawning), and batched
            # decode (max_num_seqs>1) breaks the block-diffusion model. See docs/validation_findings.md.
            # The prescriptive multilevel prompt + caps + _configure_fanout are retained for when a model
            # that is BOTH capable and fast (or true concurrency on a capable model) is available — set
            # openclaw.fanout.multilevel=true to re-enable. Default path = flat single-level fan-out (VERIFIED).
            fan_cfg = config.get('openclaw', {}).get('fanout', {}) or {}
            max_depth = int(fan_cfg.get('max_spawn_depth', 1))
            if bool(fan_cfg.get('multilevel', False)) and max_depth > 1 and len(steps) >= 4:
                message = _fanout_lead_message_multilevel(steps, max_depth)
            else:
                message = _fanout_lead_message(steps)
            # Don't block on the lead CLI to its hard cap: it hangs ~20 min past producing its answer
            # once subagents are spawned. Poll the trajectory and kill the group once the synthesis
            # is stable; lead_to stays the fallback cap (rc 124). r["returncode"] read as before.
            r = _run_lead_with_early_exit(
                _agent_cmd(model_ref, message, session_key, lead_to),
                session_key, lead_to, 'task_lead.log', env=env)
            raw = (RESULTS / 'task_lead.log').read_text(encoding='utf-8', errors='replace')
            text = extract_agent_text(raw)
            source = 'cli-json'
            if text is None:
                # The --local agent CLI commonly hangs past producing its answer once subagents are
                # spawned (it does not self-exit until children are reaped) → killed by timeout with
                # no --json on stdout. Recover the lead's synthesis from the live trajectory.
                text = _lead_synthesis_from_trajectory(session_key)
                source = 'trajectory' if text else 'none'
            got = text is not None
            append(out_md, f"\n## Lead synthesis (subagent fan-out)\n\n{text if got else '(no text returned)'}\n")
            step_results.append({'step': 1, 'returncode': r['returncode'], 'got_text': got,
                                 'source': source, 'chars': len(text) if got else 0})
            write_status(TASK_STATUS_PATH, 'running', {'completed_steps': 1, 'total_steps': 1})
        else:
            for i, step in enumerate(steps, 1):
                r = run(_agent_cmd(model_ref, step, session_key, per_step), f'task_step_{i}.log',
                        check=False, env=env, timeout=per_step + 60)
                raw = (RESULTS / f'task_step_{i}.log').read_text(encoding='utf-8', errors='replace')
                text = extract_agent_text(raw)
                got = text is not None
                body = text if got else '(no text returned)'
                append(out_md, f"\n## Step {i}\n\n**Prompt:** {step}\n\n{body}\n")
                step_results.append({'step': i, 'returncode': r['returncode'], 'got_text': got,
                                     'chars': len(text) if got else 0})
                write_status(TASK_STATUS_PATH, 'running', {'completed_steps': i, 'total_steps': len(steps)})

        manifest['task'] = {'mode': task_cfg.get('mode', 'research'), 'engine': 'openclaw-agent',
                            'orchestration': orchestration, 'session_key': session_key,
                            'steps': step_results, 'result_file': 'research_result.md'}
        if orchestration == 'subagent-fanout':
            # Judge the fan-out lead on whether its synthesis was captured: the CLI returncode is
            # often 124 from the known post-answer hang, while the text is recovered from the
            # trajectory — so a non-zero rc must NOT fail an otherwise-complete run.
            manifest['ok'] = bool(step_results) and all(s.get('got_text') for s in step_results)
        else:
            manifest['ok'] = bool(step_results) and all(s['returncode'] == 0 and s['got_text'] for s in step_results)
    except Exception as exc:
        manifest['ok'] = False
        manifest['error'] = repr(exc)
        append(RESULTS / 'error.log', f"[{now()}] task {exc!r}\n")
    manifest['finished_at'] = now()
    write_json(RESULTS / 'manifest.json', manifest)
    ok = bool(manifest.get('ok'))
    write_status(TASK_STATUS_PATH, 'done', {'ok': ok})
    TASK_DONE_PATH.write_text(json.dumps({'ok': ok, 'time': now()}), encoding='utf-8')
    bundle()


def task_status() -> None:
    _emit_status(TASK_STATUS_PATH, TASK_DONE_PATH, 'TASK_STATE')


def status() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    collect_environment()
    try:
        scfg = serve_cfg(load_json(CONFIG_PATH))
        run(f"curl -sS http://{scfg['host']}:{scfg['port']}/v1/models || true", 'status.log', check=False, timeout=30)
    except Exception:
        pass
    run(PATH_PREFIX + 'openclaw gateway status --json || openclaw gateway status || true', 'status.log', check=False, timeout=60)
    bundle()


def _snapshot_openclaw_state() -> None:
    """Best-effort: copy the agent session transcripts + workspace identity files into the results
    bundle so a run can be audited offline — e.g. did web_search actually fire (tool calls in the
    session JSONL)? was USER.md injected? Never copies .env/secrets; the forwarded key is read from
    process env by the provider and does not appear in transcripts."""
    try:
        dst = RESULTS / 'openclaw_state'
        ws = Path(os.path.expanduser(os.environ.get('OPENCLAW_WORKSPACE_DIR', '~/.openclaw/workspace')))
        for name in ('USER.md', 'AGENTS.md', 'MEMORY.md'):
            src = ws / name
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst / name)
        sess_root = Path(os.path.expanduser('~/.openclaw/agents'))
        if sess_root.exists():
            # Sort by MTIME (not filename — names are random UUIDs) and keep the most-recently-written
            # so the LEAD trajectory (written throughout the run) is always captured, not pushed out by
            # many child sessions (verified 2026-06-22: a 6-sub-question run's lead was missed at [-5:]).
            jsonls = sorted(sess_root.glob('*/sessions/*.jsonl'),
                            key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
            for jsonl in jsonls[-20:]:
                d = dst / 'sessions'
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(jsonl, d / jsonl.name)
    except Exception:
        pass


def bundle() -> None:
    # Atomic: build a pid-unique temp archive then os.replace it onto ZIP_PATH, so the launcher's
    # `bundle` exec and a worker's terminal bundle() can't corrupt the zip if they overlap.
    RESULTS.mkdir(parents=True, exist_ok=True)
    _snapshot_openclaw_state()
    tmp_base = str(BASE / f'._bundle_{os.getpid()}')
    archive = shutil.make_archive(tmp_base, 'zip', str(RESULTS))
    os.replace(archive, str(ZIP_PATH))
    print(f"Wrote {ZIP_PATH}")


def main() -> None:
    load_forwarded_secrets()  # allowlist secrets into env for this process (worker or action)
    # Detached worker entrypoint: `python remote_...py --worker <bootstrap|prompt|task>`.
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker = sys.argv[2]
        workers = {'bootstrap': _bootstrap_run, 'prompt': _prompt_run, 'task': _task_run}
        fn = workers.get(worker)
        if fn is None:
            raise ValueError(f'Unknown worker: {worker}')
        fn()
        return
    control = load_json(CONTROL_PATH, default={'action': 'status'})
    action = control.get('action', 'status')
    dispatch = {
        'bootstrap': bootstrap, 'bootstrap_status': bootstrap_status,
        'prompt': prompt, 'prompt_status': prompt_status,
        'task': task, 'task_status': task_status,
        'status': status, 'bundle': bundle,
    }
    fn = dispatch.get(action)
    if fn is None:
        raise ValueError(f'Unknown action: {action}')
    fn()


if __name__ == '__main__':
    main()
