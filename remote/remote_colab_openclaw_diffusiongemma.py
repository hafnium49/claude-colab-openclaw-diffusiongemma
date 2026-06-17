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
        rc, out = 124, (exc.output or '') + f"\n[timeout after {timeout}s]"
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


def oc_env(config: Dict[str, Any]) -> Dict[str, str]:
    """Resolve the OpenClaw gateway token + provider api key.

    Each `colab exec` is a fresh process, so later (detached) phases must re-derive these from
    the SAME source as configure_openclaw(): env first (a real Colab secret wins), then the
    config's openclaw.{gateway_token,vllm_api_key}, then a loopback default.
    """
    ocfg = config.get('openclaw', {})
    return {
        'OPENCLAW_GATEWAY_TOKEN': os.environ.get('OPENCLAW_GATEWAY_TOKEN') or ocfg.get('gateway_token') or 'colab-openclaw-local-token',
        'VLLM_API_KEY': os.environ.get('VLLM_API_KEY') or ocfg.get('vllm_api_key') or 'vllm-local',
    }


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
    args.extend(str(x) for x in vcfg.get('serve_args', []))
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
    raise ValueError(f'Unknown serve backend: {backend}')


def install_openclaw_bg():
    """Kick the OpenClaw npm installer off in the background; returns (proc, logfile)."""
    oc_cmd = 'curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard'
    oc_log = (RESULTS / 'openclaw_install.log').open('a', encoding='utf-8')
    oc_log.write(f"\n[{now()}] $ {oc_cmd}\n")
    oc_log.flush()
    proc = subprocess.Popen(oc_cmd, shell=True, executable='/bin/bash', stdout=oc_log, stderr=subprocess.STDOUT)
    return proc, oc_log


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
        + '--skip-daemon --skip-skills --skip-channels --skip-health --skip-ui --json'
    )
    run(onboard, 'openclaw_config.log', check=False, env=env, timeout=300)

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

    run(PATH_PREFIX + 'openclaw config file', 'openclaw_config.log', check=False, env=env, timeout=60)
    run(PATH_PREFIX + 'openclaw models list --json', 'openclaw_models.log', check=False, env=env, timeout=120)
    return {'model_ref': model_ref, 'provider_id': provider_id, 'gateway_port': gateway_port,
            'gateway_token_set': bool(env['OPENCLAW_GATEWAY_TOKEN']), 'compat_applied': applied}


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

def task() -> None:
    _launch_worker('task', TASK_STATUS_PATH, TASK_DONE_PATH, 'task_worker.log', 'TASK_LAUNCHED')


def _task_run() -> None:
    """Heavy autonomous task (deep research), executed detached on the VM.

    Runs each `steps` prompt sequentially through OpenClaw (one self-hosted-LLM call each — no
    paid API) and accumulates the answers into research_result.md. For true tool-using deep
    research, onboard WITHOUT --skip-skills so OpenClaw can use web/search skills.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = load_json(RESULTS / 'manifest.json', default={})
    try:
        write_status(TASK_STATUS_PATH, 'running')
        config = load_json(CONFIG_PATH)
        task_cfg = load_json(TASK_PATH)
        provider_id = config.get('openclaw', {}).get('provider_id', 'vllm')
        model_ref = config.get('openclaw', {}).get('model_ref') or f"{provider_id}/{config['model']['id']}"
        transport = task_cfg.get('transport', 'local')
        steps = task_cfg.get('steps') or [task_cfg.get('prompt') or task_cfg.get('topic') or 'Summarize your capabilities.']
        total_budget = int(task_cfg.get('timeout_seconds', 1800))
        per_step = int(task_cfg.get('step_timeout_seconds', max(120, total_budget // max(1, len(steps)))))
        ctx_budget = int(task_cfg.get('context_char_budget', 6000))  # cap on prior-step context fed forward
        env = oc_env(config)

        out_md = RESULTS / 'research_result.md'
        out_md.write_text(f"# Autonomous research result\n\n- Topic: {task_cfg.get('topic', '')}\n"
                          f"- Model: {model_ref}\n- Started: {now()}\n", encoding='utf-8')
        def _clip(s, n):
            s = (s or '').strip()
            return s if len(s) <= n else s[:n].rstrip() + ' …[truncated]'

        step_results = []
        transcript = []  # (i, step, answer) for steps that returned text — fed forward as context
        for i, step in enumerate(steps, 1):
            # Thread a BOUNDED transcript of prior answers into this step's prompt so later steps
            # (e.g. "synthesize the above") actually receive the above. Each prior answer is clipped
            # so the preamble stays well under the model's context window (ctx_budget chars total).
            if transcript:
                per = max(400, ctx_budget // len(transcript))
                notes = "\n\n".join(f"### Step {j}: {_clip(s_step, 200)}\n{_clip(ans, per)}"
                                    for (j, s_step, ans) in transcript)
                prompt_text = ("You are conducting multi-step research. Your earlier findings:\n\n"
                               f"{notes}\n\n---\nNow complete the next step, building on the findings "
                               f"above (treat them as \"the above\"):\n\n{step}")
            else:
                prompt_text = step
            r = run(_infer_cmd(model_ref, prompt_text, transport), f'task_step_{i}.log',
                    check=False, env=env, timeout=per_step)
            raw = (RESULTS / f'task_step_{i}.log').read_text(encoding='utf-8', errors='replace')
            text = extract_infer_text(raw)
            got = text is not None
            body = text if got else '(no text returned)'
            append(out_md, f"\n## Step {i}\n\n**Prompt:** {step}\n\n{body}\n")
            if got:
                transcript.append((i, step, text))
            step_results.append({'step': i, 'returncode': r['returncode'], 'got_text': got,
                                 'chars': len(text) if got else 0})
            write_status(TASK_STATUS_PATH, 'running', {'completed_steps': i, 'total_steps': len(steps)})

        manifest['task'] = {'mode': task_cfg.get('mode', 'research'), 'transport': transport,
                            'steps': step_results, 'result_file': 'research_result.md'}
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


def bundle() -> None:
    # Atomic: build a pid-unique temp archive then os.replace it onto ZIP_PATH, so the launcher's
    # `bundle` exec and a worker's terminal bundle() can't corrupt the zip if they overlap.
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp_base = str(BASE / f'._bundle_{os.getpid()}')
    archive = shutil.make_archive(tmp_base, 'zip', str(RESULTS))
    os.replace(archive, str(ZIP_PATH))
    print(f"Wrote {ZIP_PATH}")


def main() -> None:
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
