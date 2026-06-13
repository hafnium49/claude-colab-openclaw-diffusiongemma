# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Claude Code-native scaffold that controls a single Google Colab GPU runtime through the **Google Colab CLI** (`colab`). Inside that one Colab runtime it stands up a vLLM OpenAI-compatible server for `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`, an OpenClaw Gateway pointed at that local vLLM endpoint, runs a headless prompt through OpenClaw's inference CLI, and downloads a single result zip. The local machine is only the controller; the Colab runtime is a temporary, ephemeral job executor. Everything runs on loopback inside Colab — no public OpenClaw tunnel in the default workflow.

This is pure stdlib Python + bash. Locally it needs only `python` and the `colab` CLI (`pip install google-colab-cli`); vLLM and OpenClaw are installed *inside* Colab at runtime.

## Commands

```bash
# Validate the bundle locally — run this before any Colab run, and after editing
# the remote script / launcher. This IS the test suite (no pytest, no linter).
python scripts/self_test.py        # checks required files exist, JSON parses, .py compiles, bash -n
bash -n bin/colab_openclaw_diffusiongemma.sh

# Smoke test first (Qwen 0.5B on T4) — validates the orchestration path cheaply
bash bin/colab_openclaw_diffusiongemma.sh \
  --session openclaw-dg-smoke --gpu T4 \
  --config configs/smoke_test_tiny.json \
  --task examples/prompt_task.json --out ./runs/smoke

# Full quantized DiffusionGemma (needs a high-memory GPU; L4 minimum, A100/H100 preferred)
bash bin/colab_openclaw_diffusiongemma.sh \
  --session openclaw-dg --gpu L4 \
  --config configs/diffusiongemma_nvfp4.json \
  --task examples/prompt_task.json --out ./runs/openclaw-dg
```

`--keep-session` leaves the Colab runtime up for inspection (default tears it down after artifact download). The launcher runs `scripts/self_test.py` automatically before provisioning, so a self-test failure aborts the run.

There is no test framework beyond `self_test.py`. **If you add a file the workflow depends on, add it to the `required` list in `scripts/self_test.py`** or the self-test gate will not cover it.

## Architecture

Two sides connected only by the `colab` CLI as transport:

- **Local controller** — `bin/colab_openclaw_diffusiongemma.sh`. Parses flags, runs the self-test, then drives the Colab session via `colab new / status / upload / exec / download / log / stop` (see `docs/colab_cli_contract.md`). It uploads three things to `/content`: the remote orchestrator, the config JSON (`ocdg_config.json`), and the task JSON (`ocdg_task.json`). Local logs and artifacts land in `--out`.

- **Remote appliance** — `remote/remote_colab_openclaw_diffusiongemma.py`, executed inside the Colab VM. It reads its inputs from `/content/ocdg_*.json` and writes all artifacts under `/content/ocdg_results/`, then zips them to `/content/openclaw_diffusiongemma_results.zip`.

### The key pattern: control-file phase dispatch

The remote script is uploaded **once** but executed **multiple times**. `colab exec` cannot pass arguments and keeps no state between calls, so the launcher drives a multi-phase sequence by re-uploading a tiny `/content/ocdg_control.json` (`{"action": "..."}`) before each `colab exec`, and the remote `main()` branches on that action. The launcher's fixed sequence is:

1. `bootstrap` — collect environment, install + start vLLM (`nohup`, background; poll `127.0.0.1:8000/v1/models` until ready), install OpenClaw, `openclaw onboard` non-interactively, register vLLM as the `vllm` provider via `openclaw config set`, start the gateway in the background. Writes `manifest.json`, then bundles.
2. `prompt` — `openclaw infer model run --gateway --model vllm/<model_id> --prompt … --json`, salvage JSON from the output, update manifest, bundle.
3. `bundle` — re-zip `ocdg_results/` into the result archive.

A `status` action also exists for ad-hoc health checks but is not part of the launcher's default flow.

`colab exec` actually runs `remote/colab_exec_stub.py`, a two-line shim that `runpy.run_path`s the uploaded `/content/remote_colab_openclaw_diffusiongemma.py` as `__main__`. So when changing remote behavior, edit `remote_colab_openclaw_diffusiongemma.py`, not the stub.

### Config and task contract

- **Config JSON** (`configs/*.json`) drives everything model-side: `model.id`, vLLM `serve_args` / install policy / timeouts / port, and OpenClaw gateway port + token. `serve_args` is passed verbatim to `vllm serve` — the DiffusionGemma config relies on diffusion-specific flags (`--diffusion-config`, `--hf-overrides`, `--generation-config vllm`) that the smoke config omits.
- **Task JSON** (`examples/prompt_task.json`) drives the request: `prompt`, `transport` (`gateway` | `local`), `timeout_seconds`.

### Conventions & gotchas

- **Loopback by default.** vLLM `127.0.0.1:8000/v1`, gateway loopback on port `18789`. Do not add public tunnels unless the user explicitly accepts the risk.
- **Secrets via env only**, never in config files or commits: the remote forwards `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` into vLLM and `OPENCLAW_GATEWAY_TOKEN` / `VLLM_API_KEY` into OpenClaw if present in the Colab environment.
- **Failures are captured, not fatal.** Most remote steps run with `check=False` and the run always ends by bundling — so even a failed DiffusionGemma load yields a result zip with `manifest.json` + logs (`vllm.log`, `openclaw_gateway.log`, `install.log`, `error.log`). When diagnosing, read the manifest and these logs rather than assuming success/failure from the exit code.
- **Colab is ephemeral** — treat every run as batch-style; never promise durability beyond the active session.
- The `*:Zone.Identifier` files are Windows/WSL alternate-data-stream artifacts (gitignored via `*:Zone.Identifier`) and can be ignored.

## Claude Code integration

The project ships its own subagent and skill (these are the intended entry points, and `self_test.py` requires them to exist):

- `.claude/agents/colab-openclaw-diffusiongemma.md` — operator subagent with the operating rules above.
- `.claude/skills/colab-openclaw-diffusiongemma/SKILL.md` — invoke with `/colab-openclaw-diffusiongemma run configs/diffusiongemma_nvfp4.json examples/prompt_task.json`.
