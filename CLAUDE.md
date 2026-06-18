# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Claude Code-native scaffold that controls a single Google Colab GPU runtime through the **Google Colab CLI** (`colab`). Inside that one Colab runtime it stands up a **self-hosted, OpenAI-compatible LLM** (config-driven backend), points an **OpenClaw** Gateway at it on loopback, runs a headless prompt **or an autonomous multi-step task** (e.g. deep research) through OpenClaw's inference CLI, and downloads a single result zip. The **default backend is `llama_cpp`** serving **LFM2.5-8B-A1B** (4-bit GGUF) — the validated, **fee-free** path on a free Colab **T4** (~134 tok/s; vLLM can't serve ≥3B there; see `docs/t4_llama_cpp_serving.md`). For **deep research with live web search**, the **`ollama`** backend (`configs/lfm2_ollama_web.json`) serves LFM2.5 so OpenClaw gets structured `tool_calls` and actually runs `web_search`/`web_fetch` against Brave (validated 2026-06-18). (Qwen3.5-9B was the old default but its hybrid-SSM build crashes llama.cpp on a T4.) The original `vllm` backend for `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` is kept for an **L4** (`--gpu L4 --config configs/diffusiongemma_nvfp4.json`). The local machine is only the controller; the Colab runtime is a temporary, ephemeral job executor. Everything runs on loopback inside Colab — no public OpenClaw tunnel in the default workflow.

This is pure stdlib Python + bash. Locally it needs only `python` and the `colab` CLI (`pip install google-colab-cli`); the LLM backend (llama.cpp or vLLM) and OpenClaw are installed *inside* Colab at runtime.

## Commands

```bash
# Validate the bundle locally — run this before any Colab run, and after editing
# the remote script / launcher. This IS the test suite (no pytest, no linter).
python scripts/self_test.py        # checks required files exist, JSON parses, .py compiles, bash -n
bash -n bin/colab_openclaw_diffusiongemma.sh

# Cheap orchestration smoke (0.5B GGUF on T4 via llama.cpp) — validates the path fast
bash bin/colab_openclaw_diffusiongemma.sh --gpu T4 \
  --config configs/llama_smoke.json --task examples/prompt_task.json --out ./runs/smoke

# Validated default: LFM2.5-8B-A1B (4-bit GGUF) on a T4, single prompt (fee-free, self-hosted)
bash bin/colab_openclaw_diffusiongemma.sh --gpu T4 \
  --config configs/llama_lfm2.json --task examples/prompt_task.json --out ./runs/lfm2

# Deep research with LIVE web search (Ollama backend -> structured tool_calls -> real Brave search).
# Validated 2026-06-18: agent runs web_search/web_fetch, returns cited URLs, remembers the user.
# Needs BRAVE_API_KEY in ~/.env (the launcher forwards it into Colab).
bash bin/colab_openclaw_diffusiongemma.sh --gpu T4 \
  --config configs/lfm2_ollama_web.json --task examples/web_verify_task.json --out ./runs/research

# Autonomous, human-free deep-research run (detached + polled multi-step task)
bash bin/colab_openclaw_diffusiongemma.sh --gpu T4 \
  --config configs/llama_lfm2.json --task examples/research_task.json --out ./runs/research

# Original DiffusionGemma target (vLLM backend; needs an L4 entitlement)
bash bin/colab_openclaw_diffusiongemma.sh --gpu L4 \
  --config configs/diffusiongemma_nvfp4.json --task examples/prompt_task.json --out ./runs/openclaw-dg
```

`--keep-session` leaves the Colab runtime up for inspection (default tears it down after artifact download). The launcher runs `scripts/self_test.py` automatically before provisioning, so a self-test failure aborts the run.

There is no test framework beyond `self_test.py`. **If you add a file the workflow depends on, add it to the `required` list in `scripts/self_test.py`** or the self-test gate will not cover it.

## Architecture

Two sides connected only by the `colab` CLI as transport:

- **Local controller** — `bin/colab_openclaw_diffusiongemma.sh`. Parses flags, runs the self-test, then drives the Colab session via `colab new / status / upload / exec / download / log / stop` (see `docs/colab_cli_contract.md`). It uploads three things to `/content`: the remote orchestrator, the config JSON (`ocdg_config.json`), and the task JSON (`ocdg_task.json`). Local logs and artifacts land in `--out`.

- **Remote appliance** — `remote/remote_colab_openclaw_diffusiongemma.py`, executed inside the Colab VM. It reads its inputs from `/content/ocdg_*.json` and writes all artifacts under `/content/ocdg_results/`, then zips them to `/content/openclaw_diffusiongemma_results.zip`.

### The key pattern: control-file phase dispatch

The remote script is uploaded **once** but executed **multiple times**. `colab exec` cannot pass arguments and keeps no state between calls, so the launcher drives a multi-phase sequence by re-uploading a tiny `/content/ocdg_control.json` (`{"action": "..."}`) before each `colab exec`, and the remote `main()` branches on that action. **Every heavy phase runs DETACHED** (`subprocess.Popen(..., '--worker', X, start_new_session=True)`) and is polled via short `*_status` execs, so no single `colab exec` is held open through a multi-minute step (which would hit the proven ~10.5-min websocket drop). The launcher's sequence:

1. `bootstrap` (worker) — collect env, **install + start the serve backend** (config-driven: `install_llama_cpp`/`start_llama_cpp` or `install_vllm`/`start_vllm`; poll `127.0.0.1:8000/v1/models` until ready), install OpenClaw, `openclaw onboard` non-interactively, apply the **compat infer-fixes** (`openclaw config set 'models.providers.<id>.models[0]....'`), start the gateway. Writes `manifest.json` + `bootstrap.done`, then bundles. The launcher polls `bootstrap_status` (`BOOTSTRAP_STATE=ready|failed`, budget derived from the config's own timeouts) and **only infers if ready**.
2. `prompt` (worker) — one `openclaw infer model run … --json`, polled via `prompt_status`. For `task.mode == "research"` the launcher instead runs the `task` worker (sequential multi-step deep research → `research_result.md`), polled via `task_status`.
3. `bundle` — atomically re-zip `ocdg_results/` (temp + `os.replace`).

A `status` action also exists for ad-hoc health checks but is not part of the default flow.

`colab exec` actually runs `remote/colab_exec_stub.py`, a two-line shim that `runpy.run_path`s the uploaded `/content/remote_colab_openclaw_diffusiongemma.py` as `__main__`. So when changing remote behavior, edit `remote_colab_openclaw_diffusiongemma.py`, not the stub.

### Config and task contract

- **Config JSON** (`configs/*.json`) drives everything model-side via a `serve` block: `serve.backend` (`llama_cpp` | `vllm` | `ollama`), `serve.host/port/startup_timeout_seconds`, and a backend sub-block — `serve.llama_cpp.{wheel, wheel_index, server_args}` with `model.{id, gguf_repo, gguf_file}`, **or** legacy top-level `vllm.{serve_args, install_command, …}` (the DiffusionGemma config's `serve_args` carry diffusion-specific flags `--diffusion-config`/`--hf-overrides`/`--generation-config vllm`). `openclaw.compat.{requiresStringContent, supportsTools, maxTokens, contextWindow}` supplies the infer-fixes. For the `ollama` backend: `serve.ollama.{num_ctx, install_timeout_seconds, download_timeout_seconds}` with `model.id` an Ollama tag (e.g. `lfm2.5:8b`). The web-search/identity wiring is config-driven too: `openclaw.web.{enabled, provider, plugin_package, max_results, code_mode, lean_workspace}` (installs the brave plugin, enables web tools, trims the workspace) and `openclaw.identity.{name, email?}` (seeds `~/.openclaw/workspace/USER.md`). Configs: `llama_lfm2.json` (validated T4 default — LFM2.5-8B-A1B), `lfm2_ollama_web.json` (T4 deep-research with live Brave web search via `ollama`), `llama_qwen9b.json` (alt; Qwen3.5-9B is unstable under llama.cpp on T4), `llama_smoke.json` (cheap 0.5B), `diffusiongemma_nvfp4.json` (vLLM/L4), `smoke_test_tiny.json` (vLLM 0.5B).
- **Task JSON** drives the request: `prompt`, `transport` (`gateway` | `local` — `local` = direct infer, no `--gateway`, the robust path), `timeout_seconds`. For autonomous runs set `mode: "research"` with a `steps` list (+ `topic`, optional `step_timeout_seconds`); see `examples/research_task.json`.

### Conventions & gotchas

- **Loopback by default.** serve backend on `127.0.0.1:8000/v1` (**always `:8000`, never `:8080` — Colab's own node service owns 8080**), gateway loopback on port `18789`. Do not add public tunnels unless the user explicitly accepts the risk.
- **Secrets via env only**, never in config files or commits: the remote forwards `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` into the serve backend (vLLM env; llama.cpp's `hf_hub_download` reads it from env) and `OPENCLAW_GATEWAY_TOKEN` / `VLLM_API_KEY` into OpenClaw if present in the Colab environment. `oc_env()` re-derives these in every detached phase (env → `openclaw.{gateway_token,vllm_api_key}` → loopback default).
- **Failures are captured, not fatal.** Most remote steps run with `check=False` and the run always ends by bundling — so even a failed model load yields a result zip with `manifest.json` + logs (`serve.log`, `openclaw_gateway.log`, `install.log`, `llama_download.log`, `error.log`). When diagnosing, read the manifest and these logs rather than assuming success/failure from the exit code.
- **Colab is ephemeral** — treat every run as batch-style; never promise durability beyond the active session.
- The `*:Zone.Identifier` files are Windows/WSL alternate-data-stream artifacts (gitignored via `*:Zone.Identifier`) and can be ignored.

## Claude Code integration

The project ships its own subagent and skill (these are the intended entry points, and `self_test.py` requires them to exist):

- `.claude/agents/colab-openclaw-diffusiongemma.md` — operator subagent with the operating rules above.
- `.claude/skills/colab-openclaw-diffusiongemma/SKILL.md` — invoke with `/colab-openclaw-diffusiongemma run configs/diffusiongemma_nvfp4.json examples/prompt_task.json`.
