#!/usr/bin/env python3
"""Remote Colab orchestrator for OpenClaw + vLLM DiffusionGemma.

This file is executed inside a Google Colab VM by `colab exec`.
It reads control/config/task files from /content and writes artifacts to
/content/ocdg_results, then builds /content/openclaw_diffusiongemma_results.zip.
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
    proc = subprocess.run(
        cmd,
        shell=True,
        executable='/bin/bash',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged_env,
        timeout=timeout,
    )
    append(log_path, proc.stdout)
    result = {'cmd': cmd, 'returncode': proc.returncode, 'log': str(log_path)}
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\nSee {log_path}")
    return result


def http_get_json(url: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer vllm-local'})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode('utf-8'))


def wait_for_url(url: str, seconds: int, log_name: str) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            data = http_get_json(url, timeout_s=5)
            write_json(RESULTS / 'vllm_models.json', data)
            return True
        except Exception as exc:
            append(RESULTS / log_name, f"[{now()}] waiting for {url}: {exc}\n")
            time.sleep(5)
    return False


def collect_environment() -> None:
    lines = []
    lines.append(f"timestamp_utc: {now()}")
    lines.append(f"python: {sys.version}")
    for cmd in ['uname -a', 'nvidia-smi', 'python -m pip --version', 'df -h /content', 'free -h']:
        p = subprocess.run(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines.append(f"\n$ {cmd}\n{p.stdout}")
    (RESULTS / 'environment.txt').write_text('\n'.join(lines), encoding='utf-8')


def install_vllm(config: Dict[str, Any]) -> None:
    vcfg = config.get('vllm', {})
    if not vcfg.get('install', True):
        return
    install_cmd = vcfg.get('install_command') or (
        'python -m pip install -U vllm --pre '
        '--extra-index-url https://wheels.vllm.ai/nightly/cu129 '
        '--extra-index-url https://download.pytorch.org/whl/cu129'
    )
    run('python -m pip install -U pip', 'install.log', check=False, timeout=1800)
    run(install_cmd, 'install.log', check=True, timeout=int(vcfg.get('install_timeout_seconds', 3600)))


def start_vllm(config: Dict[str, Any]) -> Dict[str, Any]:
    vcfg = config.get('vllm', {})
    model_id = config['model']['id']
    host = vcfg.get('host', '127.0.0.1')
    port = int(vcfg.get('port', 8000))
    max_wait = int(vcfg.get('startup_timeout_seconds', 1800))

    env_exports = {
        'VLLM_USE_V2_MODEL_RUNNER': str(vcfg.get('use_v2_model_runner', '1')),
        'HF_HUB_ENABLE_HF_TRANSFER': str(vcfg.get('hf_transfer', '1')),
    }
    if os.environ.get('HF_TOKEN'):
        env_exports['HF_TOKEN'] = os.environ['HF_TOKEN']
    if os.environ.get('HUGGING_FACE_HUB_TOKEN'):
        env_exports['HUGGING_FACE_HUB_TOKEN'] = os.environ['HUGGING_FACE_HUB_TOKEN']

    args: List[str] = ['vllm', 'serve', shlex.quote(model_id)]
    serve_args = vcfg.get('serve_args', [])
    args.extend(str(x) for x in serve_args)
    args.extend(['--host', shlex.quote(host), '--port', str(port)])
    if 'api_key' in vcfg:
        args.extend(['--api-key', shlex.quote(str(vcfg['api_key']))])

    export_prefix = ' '.join(f"export {k}={shlex.quote(v)};" for k, v in env_exports.items())
    cmd = f"{export_prefix} nohup {' '.join(args)} > {RESULTS}/vllm.log 2>&1 & echo $! > {RESULTS}/vllm.pid"
    run(cmd, 'vllm_start.log', check=True, timeout=60)
    ok = wait_for_url(f'http://{host}:{port}/v1/models', max_wait, 'vllm_start.log')
    return {'ok': ok, 'base_url': f'http://{host}:{port}/v1', 'model_id': model_id, 'pid_file': str(RESULTS / 'vllm.pid')}


def install_openclaw(config: Dict[str, Any]) -> None:
    ocfg = config.get('openclaw', {})
    if ocfg.get('install', True):
        run('curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard', 'openclaw_install.log', check=True, timeout=int(ocfg.get('install_timeout_seconds', 1800)))
    run('export PATH="$(npm prefix -g)/bin:$PATH"; openclaw --version', 'openclaw_install.log', check=False, timeout=60)


def configure_openclaw(config: Dict[str, Any], vllm_state: Dict[str, Any]) -> Dict[str, Any]:
    ocfg = config.get('openclaw', {})
    model_id = config['model']['id']
    model_ref = ocfg.get('model_ref') or f'vllm/{model_id}'
    gateway_port = int(ocfg.get('gateway_port', 18789))
    gateway_token = os.environ.get('OPENCLAW_GATEWAY_TOKEN') or ocfg.get('gateway_token') or 'colab-openclaw-local-token'
    os.environ['OPENCLAW_GATEWAY_TOKEN'] = gateway_token
    os.environ['VLLM_API_KEY'] = ocfg.get('vllm_api_key', 'vllm-local')

    env = {'OPENCLAW_GATEWAY_TOKEN': gateway_token, 'VLLM_API_KEY': os.environ['VLLM_API_KEY']}
    path_prefix = 'export PATH="$(npm prefix -g)/bin:$PATH"; '

    onboard = (
        path_prefix
        + 'openclaw onboard --non-interactive --mode local --auth-choice skip '
        + f'--gateway-port {gateway_port} --gateway-bind loopback '
        + '--gateway-auth token --gateway-token-ref-env OPENCLAW_GATEWAY_TOKEN '
        + '--skip-skills --json'
    )
    run(onboard, 'openclaw_config.log', check=False, env=env, timeout=300)

    provider_cfg = {
        'baseUrl': vllm_state['base_url'],
        'apiKey': '${VLLM_API_KEY}',
        'api': 'openai-completions',
        'timeoutSeconds': int(ocfg.get('provider_timeout_seconds', 600)),
        'models': [
            {
                'id': model_id,
                'name': config['model'].get('name', model_id),
                'reasoning': bool(config['model'].get('reasoning', True)),
                'input': config['model'].get('input', ['text']),
                'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
                'contextWindow': int(config['model'].get('context_window', 262144)),
                'maxTokens': int(config['model'].get('max_tokens', 4096)),
            }
        ],
    }
    provider_json = json.dumps(provider_cfg)
    visible_json = json.dumps({model_ref: {}})
    primary_json = json.dumps({'primary': model_ref})

    run(path_prefix + 'openclaw config set models.providers.vllm ' + shlex.quote(provider_json) + ' --strict-json --merge', 'openclaw_config.log', check=False, env=env, timeout=120)
    run(path_prefix + 'openclaw config set agents.defaults.models ' + shlex.quote(visible_json) + ' --strict-json --merge', 'openclaw_config.log', check=False, env=env, timeout=120)
    run(path_prefix + 'openclaw config set agents.defaults.model ' + shlex.quote(primary_json) + ' --strict-json --merge', 'openclaw_config.log', check=False, env=env, timeout=120)
    run(path_prefix + 'openclaw models list --provider vllm --json', 'openclaw_models.log', check=False, env=env, timeout=120)

    return {'model_ref': model_ref, 'gateway_port': gateway_port, 'gateway_token_set': bool(gateway_token)}


def start_openclaw_gateway(config: Dict[str, Any]) -> Dict[str, Any]:
    ocfg = config.get('openclaw', {})
    gateway_port = int(ocfg.get('gateway_port', 18789))
    env = {'OPENCLAW_GATEWAY_TOKEN': os.environ.get('OPENCLAW_GATEWAY_TOKEN', 'colab-openclaw-local-token'), 'VLLM_API_KEY': os.environ.get('VLLM_API_KEY', 'vllm-local')}
    cmd = 'export PATH="$(npm prefix -g)/bin:$PATH"; nohup openclaw gateway run > ' + str(RESULTS / 'openclaw_gateway.log') + ' 2>&1 & echo $! > ' + str(RESULTS / 'openclaw_gateway.pid')
    run(cmd, 'openclaw_gateway_start.log', check=False, env=env, timeout=60)
    time.sleep(5)
    run('export PATH="$(npm prefix -g)/bin:$PATH"; openclaw gateway status --json || openclaw gateway status || true', 'openclaw_gateway_status.log', check=False, env=env, timeout=120)
    return {'gateway_port': gateway_port, 'pid_file': str(RESULTS / 'openclaw_gateway.pid')}


def bootstrap() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_json(CONFIG_PATH)
    manifest: Dict[str, Any] = {'started_at': now(), 'action': 'bootstrap', 'steps': []}
    try:
        collect_environment()
        install_vllm(config)
        vllm_state = start_vllm(config)
        manifest['vllm'] = vllm_state
        install_openclaw(config)
        openclaw_state = configure_openclaw(config, vllm_state)
        manifest['openclaw_config'] = openclaw_state
        gateway_state = start_openclaw_gateway(config)
        manifest['openclaw_gateway'] = gateway_state
        manifest['ok'] = bool(vllm_state.get('ok'))
    except Exception as exc:
        manifest['ok'] = False
        manifest['error'] = repr(exc)
        append(RESULTS / 'error.log', f"[{now()}] {exc!r}\n")
    manifest['finished_at'] = now()
    write_json(RESULTS / 'manifest.json', manifest)
    bundle()


def prompt() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_json(CONFIG_PATH)
    task = load_json(TASK_PATH)
    manifest = load_json(RESULTS / 'manifest.json', default={})
    prompt_text = task.get('prompt') or 'Reply with exactly: smoke-ok'
    model_id = config['model']['id']
    model_ref = config.get('openclaw', {}).get('model_ref') or f'vllm/{model_id}'
    transport = task.get('transport', 'gateway')
    env = {'OPENCLAW_GATEWAY_TOKEN': os.environ.get('OPENCLAW_GATEWAY_TOKEN', 'colab-openclaw-local-token'), 'VLLM_API_KEY': os.environ.get('VLLM_API_KEY', 'vllm-local')}
    path_prefix = 'export PATH="$(npm prefix -g)/bin:$PATH"; '
    flag = '--gateway' if transport == 'gateway' else '--local'
    cmd = path_prefix + 'openclaw infer model run ' + flag + ' --model ' + shlex.quote(model_ref) + ' --prompt ' + shlex.quote(prompt_text) + ' --json'
    result = run(cmd, 'openclaw_infer.txt', check=False, env=env, timeout=int(task.get('timeout_seconds', 900)))

    infer_txt = RESULTS / 'openclaw_infer.txt'
    raw = infer_txt.read_text(encoding='utf-8', errors='replace') if infer_txt.exists() else ''
    # Extract likely JSON from the final braces if the command echoed logs first.
    parsed = None
    first = raw.find('{')
    last = raw.rfind('}')
    if first != -1 and last != -1 and last > first:
        try:
            parsed = json.loads(raw[first:last+1])
        except Exception:
            parsed = None
    if parsed is not None:
        write_json(RESULTS / 'openclaw_infer.json', parsed)

    manifest['prompt'] = {'model_ref': model_ref, 'transport': transport, 'returncode': result['returncode'], 'parsed_json': parsed is not None}
    manifest['ok'] = bool(manifest.get('ok', True)) and result['returncode'] == 0
    manifest['finished_at'] = now()
    write_json(RESULTS / 'manifest.json', manifest)
    bundle()


def status() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    collect_environment()
    run('curl -sS http://127.0.0.1:8000/v1/models || true', 'status.log', check=False, timeout=30)
    run('export PATH="$(npm prefix -g)/bin:$PATH"; openclaw gateway status --json || openclaw gateway status || true', 'status.log', check=False, timeout=60)
    bundle()


def bundle() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix('')), 'zip', str(RESULTS))
    print(f"Wrote {ZIP_PATH}")


def main() -> None:
    control = load_json(CONTROL_PATH, default={'action': 'status'})
    action = control.get('action', 'status')
    if action == 'bootstrap':
        bootstrap()
    elif action == 'prompt':
        prompt()
    elif action == 'status':
        status()
    elif action == 'bundle':
        bundle()
    else:
        raise ValueError(f'Unknown action: {action}')


if __name__ == '__main__':
    main()
