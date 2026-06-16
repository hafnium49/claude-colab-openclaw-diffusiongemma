#!/usr/bin/env python3
"""Remote Colab orchestrator for OpenClaw + a self-hosted LLM.

Executed inside a Google Colab VM by `colab exec` (via colab_exec_stub.py). Reads
control/config/task files from /content and writes artifacts to /content/ocdg_results,
then builds /content/openclaw_diffusiongemma_results.zip.

Serving backend is CONFIG-DRIVEN (`serve.backend`):
  - "llama_cpp" (VALIDATED, fee-free): prebuilt llama-cpp-python[server] CUDA wheel serves a
    local GGUF on loopback. This is the only backend that serves >=3B on a Colab T4 (vLLM's
    FlashInfer crashes on Turing/sm_75). See docs/t4_llama_cpp_serving.md.
  - "vllm": kept for the original DiffusionGemma/L4 target (old configs without a `serve` block
    fall back to this via their top-level `vllm` section).

Phases are driven by /content/ocdg_control.json `{"action": ...}`, re-uploaded by the launcher
before each exec (the kernel keeps no state between execs). Long work (install/serve, and the
autonomous task) runs DETACHED and is polled via short *_status execs, so no single exec is held
open through a multi-minute step.
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
STATUS_PATH = RESULTS / 'bootstrap_status.json'
DONE_PATH = RESULTS / 'bootstrap.done'
TASK_STATUS_PATH = RESULTS / 'task_status.json'
TASK_DONE_PATH = RESULTS / 'task.done'
SELF_PATH = BASE / 'remote_colab_openclaw_diffusiongemma.py'
GGUF_DIR = BASE / 'gguf'

# Resolve the openclaw binary by absolute path -> never 'openclaw: command not found' regardless
# of PATH quirks. (The npm installer symlinks it into the global bin, e.g. /usr/bin/openclaw.)
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
    """Salvage the model text from an `openclaw infer ... --json` log (it may echo logs first)."""
    first, last = raw.find('{'), raw.rfind('}')
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        parsed = json.loads(raw[first:last + 1])
    except Exception:
        return None
    try:
        return parsed['outputs'][0]['text']
    except Exception:
        return json.dumps(parsed)


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
    # Legacy: old configs put everything under a top-level `vllm` block.
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


def install_vllm(config: Dict[str, Any], scfg: Dict[str, Any]) -> None:
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


def install_llama_cpp(config: Dict[str, Any], scfg: Dict[str, Any]) -> None:
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

    # Download the GGUF (public repos need no token; hf_hub reads HF_TOKEN from env if present).
    dl = (f"{sys.executable} -c \"from huggingface_hub import hf_hub_download as d; "
          f"print(d({repo!r}, {gguf_file!r}, local_dir={str(GGUF_DIR)!r}))\"")
    run(dl, 'llama_download.log', check=True, timeout=int(lcfg.get('download_timeout_seconds', 1800)))
    gguf_path = str(GGUF_DIR / gguf_file)

    args = [sys.executable, '-m', 'llama_cpp.server', '--model', shlex.quote(gguf_path),
            '--model_alias', shlex.quote(model_id)] + server_args + \
           ['--host', shlex.quote(host), '--port', str(port)]
    cmd = f"nohup {' '.join(args)} > {RESULTS}/serve.log 2>&1 & echo $! > {RESULTS}/serve.pid"
    run(cmd, 'serve_start.log', check=True, timeout=60)
    ok = wait_for_url(f'http://{host}:{port}/v1/models', max_wait, 'serve_start.log')
    return {'ok': ok, 'base_url': f'http://{host}:{port}/v1', 'model_id': model_id, 'backend': 'llama_cpp'}


def start_backend(config: Dict[str, Any]) -> Dict[str, Any]:
    scfg = serve_cfg(config)
    backend = scfg['backend']
    if backend == 'llama_cpp':
        install_llama_cpp(config, scfg)
        return start_llama_cpp(config, scfg)
    if backend == 'vllm':
        install_vllm(config, scfg)
        return start_vllm(config, scfg)
    raise ValueError(f'Unknown serve backend: {backend}')


def install_openclaw_bg() -> Any:
    """Kick the OpenClaw npm installer off in the background; returns (proc, logfile) or (None, None)."""
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
    gateway_token = os.environ.get('OPENCLAW_GATEWAY_TOKEN') or ocfg.get('gateway_token') or 'colab-openclaw-local-token'
    os.environ['OPENCLAW_GATEWAY_TOKEN'] = gateway_token
    os.environ['VLLM_API_KEY'] = ocfg.get('vllm_api_key', 'vllm-local')
    env = {'OPENCLAW_GATEWAY_TOKEN': gateway_token, 'VLLM_API_KEY': os.environ['VLLM_API_KEY']}

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
    # `models.providers.<id>.models[0]...` index form is valid ([] errors). Applied only
    # when the config supplies a `compat` block (legacy DiffusionGemma config omits it).
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
            'gateway_token_set': bool(gateway_token), 'compat_applied': applied}


def start_openclaw_gateway(config: Dict[str, Any]) -> Dict[str, Any]:
    ocfg = config.get('openclaw', {})
    gateway_port = int(ocfg.get('gateway_port', 18789))
    env = {'OPENCLAW_GATEWAY_TOKEN': os.environ.get('OPENCLAW_GATEWAY_TOKEN', 'colab-openclaw-local-token'),
           'VLLM_API_KEY': os.environ.get('VLLM_API_KEY', 'vllm-local')}
    cmd = PATH_PREFIX + 'setsid nohup openclaw gateway run > ' + str(RESULTS / 'openclaw_gateway.log') + \
          ' 2>&1 < /dev/null & echo $! > ' + str(RESULTS / 'openclaw_gateway.pid')
    run(cmd, 'openclaw_gateway_start.log', check=False, env=env, timeout=60)
    time.sleep(8)
    run(PATH_PREFIX + 'openclaw gateway status --json || openclaw gateway status || true',
        'openclaw_gateway_status.log', check=False, env=env, timeout=120)
    return {'gateway_port': gateway_port, 'pid_file': str(RESULTS / 'openclaw_gateway.pid')}


# ---------------------------------------------------------------------------
# Phase: bootstrap (serve backend + onboard OpenClaw), detached + polled
# ---------------------------------------------------------------------------

def bootstrap() -> None:
    """Fast action: launch the heavy bootstrap DETACHED and return immediately."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    for marker in (DONE_PATH, STATUS_PATH):
        if marker.exists():
            marker.unlink()
    write_status(STATUS_PATH, 'launching')
    worker_log = (RESULTS / 'bootstrap_worker.log').open('a', encoding='utf-8')
    subprocess.Popen([sys.executable, str(SELF_PATH), '--worker', 'bootstrap'],
                     stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True)
    print('BOOTSTRAP_LAUNCHED')


def _bootstrap_run() -> None:
    """Heavy bootstrap, executed detached on the VM (not inside a colab exec)."""
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
        append(RESULTS / 'error.log', f"[{now()}] {exc!r}\n")
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
    """Fast action: report bootstrap progress for the launcher's poll loop."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    detail: Dict[str, Any] = {}
    if DONE_PATH.exists():
        try:
            done = json.loads(DONE_PATH.read_text(encoding='utf-8'))
        except Exception:
            done = {'ok': False}
        detail['done'] = done
        state = 'ready' if done.get('ok') else 'failed'
    else:
        if STATUS_PATH.exists():
            try:
                detail['status'] = json.loads(STATUS_PATH.read_text(encoding='utf-8'))
            except Exception:
                pass
        state = 'running'
    try:
        detail['serve_up'] = _probe_serve_up(load_json(CONFIG_PATH))
    except Exception:
        detail['serve_up'] = False
    print('BOOTSTRAP_STATE=' + state)
    print('BOOTSTRAP_DETAIL=' + json.dumps(detail))


# ---------------------------------------------------------------------------
# Phase: prompt (single synchronous infer — smoke test)
# ---------------------------------------------------------------------------

def _infer_cmd(model_ref: str, prompt_text: str, transport: str) -> str:
    # transport 'gateway' -> --gateway; anything else -> direct infer (no flag), the robust path.
    flag = '--gateway ' if transport == 'gateway' else ''
    return (PATH_PREFIX + 'openclaw infer model run ' + flag
            + '--model ' + shlex.quote(model_ref) + ' --prompt ' + shlex.quote(prompt_text) + ' --json')


def prompt() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_json(CONFIG_PATH)
    task = load_json(TASK_PATH)
    manifest = load_json(RESULTS / 'manifest.json', default={})
    prompt_text = task.get('prompt') or 'Reply with exactly: smoke-ok'
    model_id = config['model']['id']
    provider_id = config.get('openclaw', {}).get('provider_id', 'vllm')
    model_ref = config.get('openclaw', {}).get('model_ref') or f'{provider_id}/{model_id}'
    transport = task.get('transport', 'gateway')
    env = {'OPENCLAW_GATEWAY_TOKEN': os.environ.get('OPENCLAW_GATEWAY_TOKEN', 'colab-openclaw-local-token'),
           'VLLM_API_KEY': os.environ.get('VLLM_API_KEY', 'vllm-local')}
    result = run(_infer_cmd(model_ref, prompt_text, transport), 'openclaw_infer.txt',
                 check=False, env=env, timeout=int(task.get('timeout_seconds', 900)))

    raw = (RESULTS / 'openclaw_infer.txt').read_text(encoding='utf-8', errors='replace')
    text = extract_infer_text(raw)
    if text is not None:
        write_json(RESULTS / 'openclaw_infer.json', {'text': text})
    manifest['prompt'] = {'model_ref': model_ref, 'transport': transport,
                          'returncode': result['returncode'], 'got_text': text is not None}
    manifest['ok'] = bool(manifest.get('ok', True)) and result['returncode'] == 0
    manifest['finished_at'] = now()
    write_json(RESULTS / 'manifest.json', manifest)
    bundle()


# ---------------------------------------------------------------------------
# Phase: task (autonomous, time-consuming job — deep research), detached + polled
# ---------------------------------------------------------------------------

def task() -> None:
    """Fast action: launch the autonomous task worker DETACHED and return immediately."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    for marker in (TASK_DONE_PATH, TASK_STATUS_PATH):
        if marker.exists():
            marker.unlink()
    write_status(TASK_STATUS_PATH, 'launching')
    worker_log = (RESULTS / 'task_worker.log').open('a', encoding='utf-8')
    subprocess.Popen([sys.executable, str(SELF_PATH), '--worker', 'task'],
                     stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True)
    print('TASK_LAUNCHED')


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
        env = {'OPENCLAW_GATEWAY_TOKEN': os.environ.get('OPENCLAW_GATEWAY_TOKEN', 'colab-openclaw-local-token'),
               'VLLM_API_KEY': os.environ.get('VLLM_API_KEY', 'vllm-local')}

        out_md = RESULTS / 'research_result.md'
        out_md.write_text(f"# Autonomous research result\n\n- Topic: {task_cfg.get('topic', '')}\n"
                          f"- Model: {model_ref}\n- Started: {now()}\n", encoding='utf-8')
        step_results = []
        for i, step in enumerate(steps, 1):
            r = run(_infer_cmd(model_ref, step, transport), f'task_step_{i}.log',
                    check=False, env=env, timeout=per_step)
            raw = (RESULTS / f'task_step_{i}.log').read_text(encoding='utf-8', errors='replace')
            text = extract_infer_text(raw) or '(no text returned)'
            append(out_md, f"\n## Step {i}\n\n**Prompt:** {step}\n\n{text}\n")
            step_results.append({'step': i, 'returncode': r['returncode'], 'chars': len(text)})
            write_status(TASK_STATUS_PATH, 'running', {'completed_steps': i, 'total_steps': len(steps)})

        manifest['task'] = {'mode': task_cfg.get('mode', 'research'), 'transport': transport,
                            'steps': step_results, 'result_file': 'research_result.md'}
        manifest['ok'] = bool(step_results) and all(s['returncode'] == 0 for s in step_results)
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
    """Fast action: report autonomous-task progress for the launcher's poll loop."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    detail: Dict[str, Any] = {}
    if TASK_DONE_PATH.exists():
        try:
            done = json.loads(TASK_DONE_PATH.read_text(encoding='utf-8'))
        except Exception:
            done = {'ok': False}
        detail['done'] = done
        state = 'ready' if done.get('ok') else 'failed'
    else:
        if TASK_STATUS_PATH.exists():
            try:
                detail['status'] = json.loads(TASK_STATUS_PATH.read_text(encoding='utf-8'))
            except Exception:
                pass
        state = 'running'
    print('TASK_STATE=' + state)
    print('TASK_DETAIL=' + json.dumps(detail))


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
    RESULTS.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix('')), 'zip', str(RESULTS))
    print(f"Wrote {ZIP_PATH}")


def main() -> None:
    # Detached worker entrypoint: `python remote_...py --worker <bootstrap|task>`.
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker = sys.argv[2]
        if worker == 'bootstrap':
            _bootstrap_run()
        elif worker == 'task':
            _task_run()
        else:
            raise ValueError(f'Unknown worker: {worker}')
        return
    control = load_json(CONTROL_PATH, default={'action': 'status'})
    action = control.get('action', 'status')
    dispatch = {
        'bootstrap': bootstrap, 'bootstrap_status': bootstrap_status,
        'task': task, 'task_status': task_status,
        'prompt': prompt, 'status': status, 'bundle': bundle,
    }
    fn = dispatch.get(action)
    if fn is None:
        raise ValueError(f'Unknown action: {action}')
    fn()


if __name__ == '__main__':
    main()
