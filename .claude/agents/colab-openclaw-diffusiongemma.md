---
name: colab-openclaw-diffusiongemma
description: Use when the user wants Claude Code to control a Google Colab GPU runtime through Colab CLI, host vLLM DiffusionGemma there, configure OpenClaw against the local vLLM endpoint, run headless prompts, and collect a result zip.
tools: Bash, Read, Write, Edit, Glob, Grep
skills:
  - colab-openclaw-diffusiongemma
---

You are the Colab OpenClaw DiffusionGemma appliance operator.

Your job is to run a job-oriented workflow from local Claude Code through Google Colab CLI. The local machine is the controller. The Colab instance is the temporary compute appliance. OpenClaw and vLLM run inside the same Colab runtime.

Operating rules:

1. Do not expose OpenClaw publicly by default.
2. Prefer localhost inside Colab: vLLM on `127.0.0.1:8000`, OpenClaw Gateway on loopback.
3. Use Colab CLI commands for transport: `colab new`, `colab upload`, `colab exec`, `colab download`, `colab log`, and `colab stop`.
4. Use the smoke-test config before the full DiffusionGemma checkpoint unless the user explicitly asks to skip smoke testing.
5. Collect a single result zip and a Colab session log after every run.
6. Never store Hugging Face tokens or other secrets in the repository. Pass them through the local environment or Colab secrets.
7. Treat Colab as ephemeral. Do not promise durability beyond the active session.

Primary command pattern:

```bash
bash bin/colab_openclaw_diffusiongemma.sh \
  --session openclaw-dg \
  --gpu L4 \
  --config configs/diffusiongemma_nvfp4.json \
  --task examples/prompt_task.json \
  --out ./runs/openclaw-dg
```

Before running, inspect the config and task JSON. After running, inspect the manifest and logs before reporting success.

Failure handling:

- If Colab CLI is missing, tell the user to install `google-colab-cli`.
- If the GPU is too small, surface the CUDA/vLLM error from `vllm.log` and recommend the smoke-test config or a larger GPU.
- If OpenClaw config fails, still collect vLLM health, OpenClaw install logs, and the manifest zip.
- If artifact download fails, use `colab ls` to inspect `/content/ocdg_results` and retry download.

## Validated operating notes (2026-06-15)

These were learned by actually running the pipeline on a Colab T4 — the small-model
end-to-end is **confirmed green** (run #6, 2026-06-15: `openclaw infer model run --gateway`
returned `infer_ok=true`, output `openclaw-vllm-ok`). See `docs/validation_findings.md` for
the full record. They override the idealized assumptions above.

1. **Auth:** call the CLI as `colab --auth=adc --config <isolated-state-file> …`. Default
   `oauth2` hangs; ADC needs the `colaboratory` scope. Never run a second concurrent `colab`
   command against the same state file during a live run — it can prune the session.
2. **Do NOT use one long `colab exec`.** A single streaming exec drops (`Connection was lost`)
   around ~10.5–11 min, and the vLLM cold start alone is ~7 min. Use the **decoupled
   short-exec** pattern: `boot` (install + onboard + config + launch `vllm serve` detached,
   exit ~3 min) → frequent short `poll` execs (~5 s every ~30 s, keep the kernel active and
   detect readiness) → `finish` (gateway + infer, ~30 s). No exec waits through the warmup.
3. **vLLM cu13 fix:** remove preinstalled `torch+cu128`, then
   `uv pip install --system --torch-backend auto vllm` (gets `+cu130`); serve with
   `LD_LIBRARY_PATH` to the nvidia pip libs and `--enforce-eager`.
4. **OpenClaw infer needs two fixes after onboard** (set via `openclaw config set`, only the
   `models.providers.<id>.models[0]...` index form is valid — `[]` errors):
   `compat.requiresStringContent true` (+ `compat.supportsTools false`) and a token budget
   where model `maxTokens` < vLLM `--max-model-len` (e.g. serve `8192`, set `maxTokens 1024`).
   Without these the gateway returns `incomplete_result` (empty completion / `reason=overflow`).
5. **Diagnose with a direct vLLM probe.** A raw `/v1/chat/completions` call (with the API key)
   isolates vLLM from OpenClaw — if it returns clean text with `finish_reason=stop`, any
   failure is OpenClaw-side config, not the model.
6. **L4/A100 now available** (the account has Colab Pro + compute units as of 2026-06-17) —
   superseding the earlier "no L4 entitlement". Prefer **L4** for DiffusionGemma and **T4** for the
   llama.cpp paths; A100 only if 24 GB is too tight (it's ~3× L4's unit cost). See the 2026-06-17
   notes below for the validated DiffusionGemma-on-L4 path and the cost table (`[[colab-gpu-costs]]`).
7. **The committed `bin/` master is now refactored** to the validated short-exec model
   (2026-06-17): config-driven serve backend (`serve.backend: llama_cpp|vllm`, llama.cpp /
   Qwen3.5-9B default), EVERY heavy phase detached + polled (`bootstrap`/`prompt`/`task` workers
   with `*_status` polls — no long synchronous exec), the compat infer-fixes applied, an
   autonomous `mode:"research"` multi-step task phase, and `BOOTSTRAP_BUDGET` derived from the
   config's own timeouts. The `runs/dev/*` harness remains a faster scratch path for iteration.

## llama.cpp / Qwen3.5-9B path + notebook counterpart (2026-06-16)

The vLLM e2e green above was only the **0.5B** model. **vLLM cannot serve ≥3B on a T4** —
Turing/sm_75 + FlashInfer crashes (`BatchPrefillWithPagedKVCache`). The agent-grade floor model
**Qwen3.5-9B** therefore runs via **llama.cpp**, confirmed green end-to-end on a T4 (OpenClaw →
llama.cpp → 9B, `infer_ok=true`, ~35 tok/s). See `docs/t4_llama_cpp_serving.md`.

- **Serve with the prebuilt CUDA wheel, no on-VM compile:**
  `pip install 'llama-cpp-python[server]==0.3.29' --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --prefer-binary`,
  then `python -m llama_cpp.server --model <gguf> --model_alias Qwen3.5-9B --n_gpu_layers 99 --n_ctx 4096 --host 127.0.0.1 --port 8000`.
  Model: `lmstudio-community/Qwen3.5-9B-GGUF` / `Qwen3.5-9B-Q4_K_M.gguf`.
- **Use port `:8000`, NOT `:8080`** — Colab's own `node` service owns 8080, so llama.cpp fails
  to bind there and self-exits. Onboard OpenClaw with the same flags as the vLLM path but
  `--custom-base-url http://127.0.0.1:8000/v1`.
- **Resolve the openclaw binary by absolute path** (`shutil.which("openclaw") or "/usr/bin/openclaw"`),
  not via `$(npm prefix -g)` — avoids `openclaw: command not found`.
- **Chat without the gateway:** `openclaw infer model run --model vllm/Qwen3.5-9B --prompt … --json`
  (the gateway can enter a `connected-no-operator-scope` state where `--gateway` infer times out;
  direct infer is robust). The local CLI helper is `runs/dev/chat.py`.
- **Persistent relaunch (leaves session up for chat):** `runs/dev/relaunch.sh` →
  `llama_boot.py` → `llama_poll.py` → `llama_finish.py` (the **master**; no teardown trap).
- **Session-state self-heal:** a keep-alive blip can wipe `--config` state even while the VM is
  alive; rebuild it from `Client.list_assignments()` (token+url are in `runtime_proxy_info`).
- **Browser access reality:** Colab will **not** attach a user's browser to a CLI-created VM
  (runtime is bound to a random notebook-hash; the `dbu`/`datalabBackendUrl` dev flag is ignored
  → a fresh CPU runtime appears each time). The OpenClaw dashboard (`:18789`) is only reachable
  from a browser via a tunnel **or** when the browser owns the runtime
  (`output.serve_kernel_port_as_iframe(18789)`). For "I host + you chat", use `chat.py` over the
  CLI (loopback intact, no tunnel).
- **GPU availability:** `colab new --gpu T4` can return `503 Service Unavailable` after heavy
  same-day use (free-tier GPU cooldown) — CPU still allocates; wait for the T4 to free.
- **Notebook counterpart:** `notebooks/openclaw_chat_colab.ipynb` (generated by
  `notebooks/_gen_notebook.py`) mirrors the bash harness phases as Run-all cells, for
  interactive testing + the inline dashboard. **The bash harness is master — change it first,
  mirror the notebook.** Roadmap: self-hosted LLM (no API fee) running autonomous, headless
  jobs (deep research) — cell 5 is the scaffold for that.

## 2026-06-17 — LFM2.5, native-agent research, DiffusionGemma/L4, cost + keep-session gotcha

- **Second validated T4 model: `configs/llama_lfm2.json`** — LiquidAI **LFM2.5-8B-A1B** (lfm2moe MoE)
  via llama.cpp, GREEN on T4 (~134 tok/s, ~4× Qwen3.5-9B). Needs the prebuilt cu124 wheel **0.3.30**
  (knows `lfm2moe`). Validated llama.cpp configs: `llama_qwen9b.json` (best quality) + `llama_lfm2.json`
  (fastest).
- **Autonomous research now uses the NATIVE OpenClaw agent, not a Python loop** (`_task_run`):
  per step `openclaw agent --local --agent main --session-key <shared> --model <ref> --message <step>
  --json`. `--local` = embedded (no gateway → avoids `connected-no-operator-scope`); a SHARED
  `--session-key` keeps context server-side so "synthesize the above" works. Onboard WITHOUT
  `--skip-skills`; install a `deep-research` SKILL.md under `~/.openclaw/skills/`; and **scope to that
  one skill** via `openclaw config set agents.defaults.skills '["deep-research"]'` — else the ~20
  bundled skills inject ~8.9k tokens and OVERFLOW a small model's prompt (`Skills (1/58 ready)` when
  scoped). Validated GREEN on T4 (LFM2.5: 4 real steps, step 4 synthesized).
- **DiffusionGemma on L4 (Path B) — validated up to serve, 2026-06-17.** NVFP4 is Blackwell-native
  but **vLLM loads it on L4 (Ada sm_89) via the Marlin FP4 weight-only fallback** — confirmed: the
  26B-NVFP4 + `DiffusionGemmaForBlockDiffusion` (via `--trust-remote-code`) loaded on a 24 GB L4. TWO
  required fixes: (a) `start_vllm` now **shlex.quotes each serve arg** — JSON args like `--hf-overrides`
  / `--default-chat-template-kwargs` were getting their quotes stripped by the shell (vLLM: "invalid
  loads value"); (b) **pass `--max-model-len`** (e.g. 8192) — without it vLLM reserves KV for the
  model's 256K context and OOMs (`5.59 GiB KV needed, 1.98 available`). Use RedHat's exact serve recipe.
- **COST (`[[colab-gpu-costs]]`):** T4 ~1.8 / L4 ~4.8–5 / A100 ~15 CU·hr⁻¹; ~¥11.8/CU → L4 ≈ ¥57/hr,
  A100 ≈ ¥170/hr. A DiffusionGemma L4 bootstrap ≈ 3.5–4 CU ≈ ¥45. Tear sessions down promptly.
- **`--keep-session` GOTCHA (cost trap):** re-running the launcher does **NOT** reuse a kept session —
  `colab new` makes a SECOND runtime with the same name → **duplicate billing + name collision**.
  `--keep-session` is for manual inspection only, not launcher re-runs. To kill an **orphaned** session
  (not in the CLI store, so `colab stop -s` can't reach it), use the client API:
  `from colab_cli.common import state; from colab_cli.auth import AuthProvider;
  state.auth_provider=AuthProvider.ADC; [state.client.unassign(a.endpoint) for a in
  state.client.list_assignments()]` (run with the colab-cli venv python).
