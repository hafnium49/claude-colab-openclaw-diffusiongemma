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
STATUS_PATH = RESULTS / 'bootstrap_status.json'
DONE_PATH = RESULTS / 'bootstrap.done'
SELF_PATH = BASE / 'remote_colab_openclaw_diffusiongemma.py'


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


def write_status(stage: str, extra: Optional[Dict[str, Any]] = None) -> None:
    data: Dict[str, Any] = {'stage': stage, 'time': now()}
    if extra:
        data.update(extra)
    write_json(STATUS_PATH, data)


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
    # Ensure the CUDA runtime libs from the nvidia-* pip packages (e.g.
    # libcudart.so.13) are on the loader path, otherwise `import vllm._C` fails
    # with "libcudart.so.13: cannot open shared object file".
    ld_glob = "$(ls -d /usr/local/lib/python*/dist-packages/nvidia/*/lib 2>/dev/null | tr '\\n' ':')"
    ld_fix = 'export LD_LIBRARY_PATH="' + ld_glob + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
    cmd = f"{ld_fix}{export_prefix} nohup {' '.join(args)} > {RESULTS}/vllm.log 2>&1 & echo $! > {RESULTS}/vllm.pid"
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
    provider_id = ocfg.get('provider_id', 'vllm')
    model_ref = ocfg.get('model_ref') or f'{provider_id}/{model_id}'
    gateway_port = int(ocfg.get('gateway_port', 18789))
    gateway_token = os.environ.get('OPENCLAW_GATEWAY_TOKEN') or ocfg.get('gateway_token') or 'colab-openclaw-local-token'
    os.environ['OPENCLAW_GATEWAY_TOKEN'] = gateway_token
    os.environ['VLLM_API_KEY'] = ocfg.get('vllm_api_key', 'vllm-local')

    env = {'OPENCLAW_GATEWAY_TOKEN': gateway_token, 'VLLM_API_KEY': os.environ['VLLM_API_KEY']}
    path_prefix = 'export PATH="$(npm prefix -g)/bin:$PATH"; '

    # A single non-interactive `onboard` configures the custom vLLM provider, the
    # default model, AND the loopback token gateway in one shot. Validated live
    # against OpenClaw 2026.6.6: `--accept-risk` is REQUIRED with
    # `--non-interactive`, and `--auth-choice custom-api-key` + the `--custom-*`
    # flags write models.providers.<id> and agents.defaults.model directly — so
    # no separate `config set` calls are needed (the old `--merge` flag doesn't
    # even exist on this CLI). `--skip-daemon` keeps the gateway out of
    # systemd/launchd (unavailable in the Colab container); we start it manually.
    onboard = (
        path_prefix
        + 'openclaw onboard --non-interactive --accept-risk --mode local '
        + '--auth-choice custom-api-key '
        + f'--custom-provider-id {shlex.quote(provider_id)} '
        + f'--custom-base-url {shlex.quote(vllm_state["base_url"])} '
        + f'--custom-model-id {shlex.quote(model_id)} '
        + '--custom-compatibility openai --custom-api-key "${VLLM_API_KEY}" --custom-text-input '
        + f'--gateway-port {gateway_port} --gateway-bind loopback '
        + '--gateway-auth token --gateway-token-ref-env OPENCLAW_GATEWAY_TOKEN '
        + '--skip-daemon --skip-skills --skip-channels --skip-health --skip-ui --json'
    )
    run(onboard, 'openclaw_config.log', check=False, env=env, timeout=300)
    # Record the resulting config + model catalog for diagnostics.
    run(path_prefix + 'openclaw config file', 'openclaw_config.log', check=False, env=env, timeout=60)
    run(path_prefix + 'openclaw models list --json', 'openclaw_models.log', check=False, env=env, timeout=120)

    return {'model_ref': model_ref, 'provider_id': provider_id, 'gateway_port': gateway_port, 'gateway_token_set': bool(gateway_token)}


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
    """Fast action: launch the heavy bootstrap DETACHED and return immediately.

    A `colab exec` cannot be held open for the minutes the install takes (the
    kernel websocket drops, and without a working keep-alive the VM is reclaimed
    ~10 min in), so the real work runs as a detached worker process and the
    launcher polls `bootstrap_status` instead of blocking on one long exec.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    for marker in (DONE_PATH, STATUS_PATH):
        if marker.exists():
            marker.unlink()
    write_status('launching')
    worker_log = (RESULTS / 'bootstrap_worker.log').open('a', encoding='utf-8')
    subprocess.Popen(
        [sys.executable, str(SELF_PATH), '--worker', 'bootstrap'],
        stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    print('BOOTSTRAP_LAUNCHED')


def _bootstrap_run() -> None:
    """Heavy bootstrap, executed detached on the VM (not inside a colab exec).

    Installs vLLM and OpenClaw in parallel to fit short VM windows, starts vLLM,
    configures OpenClaw against it, and starts the gateway. Progress is written
    to bootstrap_status.json and completion (ok flag) to bootstrap.done.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {'started_at': now(), 'action': 'bootstrap', 'steps': []}
    try:
        write_status('environment')
        collect_environment()
        config = load_json(CONFIG_PATH)

        # Overlap the two slow downloads: kick OpenClaw's installer off in the
        # background, install vLLM in the foreground, then join.
        write_status('installing')
        ocfg = config.get('openclaw', {})
        oc_proc = None
        oc_log = None
        if ocfg.get('install', True):
            oc_cmd = 'curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard'
            oc_log = (RESULTS / 'openclaw_install.log').open('a', encoding='utf-8')
            oc_log.write(f"\n[{now()}] $ {oc_cmd}\n")
            oc_log.flush()
            oc_proc = subprocess.Popen(oc_cmd, shell=True, executable='/bin/bash',
                                       stdout=oc_log, stderr=subprocess.STDOUT)
        install_vllm(config)
        if oc_proc is not None:
            oc_proc.wait()
            oc_log.flush()
            oc_log.close()
        run('export PATH="$(npm prefix -g)/bin:$PATH"; openclaw --version', 'openclaw_install.log', check=False, timeout=60)

        write_status('starting_vllm')
        vllm_state = start_vllm(config)
        manifest['vllm'] = vllm_state

        write_status('configuring_openclaw')
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
    ok = bool(manifest.get('ok'))
    write_status('done', {'ok': ok})
    DONE_PATH.write_text(json.dumps({'ok': ok, 'time': now()}), encoding='utf-8')
    bundle()


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
        config = load_json(CONFIG_PATH)
        vcfg = config.get('vllm', {})
        host = vcfg.get('host', '127.0.0.1')
        port = int(vcfg.get('port', 8000))
        http_get_json(f'http://{host}:{port}/v1/models', timeout_s=3)
        detail['vllm_up'] = True
    except Exception:
        detail['vllm_up'] = False
    print('BOOTSTRAP_STATE=' + state)
    print('BOOTSTRAP_DETAIL=' + json.dumps(detail))


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
    # Detached worker entrypoint: `python remote_...py --worker bootstrap`.
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        if sys.argv[2] == 'bootstrap':
            _bootstrap_run()
            return
        raise ValueError(f'Unknown worker: {sys.argv[2]}')
    control = load_json(CONTROL_PATH, default={'action': 'status'})
    action = control.get('action', 'status')
    if action == 'bootstrap':
        bootstrap()
    elif action == 'bootstrap_status':
        bootstrap_status()
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
