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

## CRITICAL prerequisite — `colab` CLI ≥ 0.6.0 (keep-alive bug)

**Before any run, verify `colab version` reports ≥ 0.6.0** (`uv tool upgrade google-colab-cli`, or
`colab update --install`). CLIs ≤ 0.5.x use a keep-alive RPC (`RuntimeService/KeepAliveAssignment`,
hardcoded quota project `1014160490159`) that returns **`403 USER_PROJECT_DENIED`** for ordinary
external accounts. With keep-alive dead, Colab **idle-prunes the VM at ~10–12 min REGARDLESS of
kernel activity** — confirmed empirically: even a continuous-heartbeat exec (kernel busy every 5 s)
couldn't push past it, on both T4 and L4. That silently kills any run whose bootstrap exceeds ~10 min
(vLLM/DiffusionGemma cold start is ~10–40 min) — it is the real cause behind the long-standing
"~10-minute lifetime" and the "DiffusionGemma reaches serve but the run never completes" symptom.
**0.6.0 (2026-06-15) switched to a tunnel-frontend keep-alive ping** (`GET /tun/m/<endpoint>/keep-alive/`,
no project quota) that works for everyone — the VM then lives to Colab's normal limits and long
bootstraps complete. Verify the fix is live: NO `USER_PROJECT_DENIED` in `~/.config/colab-cli/colab.log`,
and a session survives past ~12 min. The launcher's `poll_worker` was also hardened (2026-06-17, on
`main`): `timeout`-wrapped status upload/exec via `$COLAB_BIN`, so a flaky kernel websocket can't hang
a poll for minutes and stall the loop past the prune. (`timeout` execs a real binary — it can't invoke
the `colab` shell function, and `timeout command colab …` fails because `command` is a builtin.)

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
   (2026-06-17): config-driven serve backend (`serve.backend: llama_cpp|vllm|ollama`, llama.cpp /
   LFM2.5-8B-A1B default — Qwen3.5-9B was the old default but crashes llama.cpp on a T4), EVERY heavy phase detached + polled (`bootstrap`/`prompt`/`task` workers
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
- **DiffusionGemma on L4 (Path B) — FULL END-TO-END GREEN, 2026-06-17** (gateway → vLLM → 26B-NVFP4
  returned a real thinking-mode completion: `ok:true`, `got_text:true`, `returncode:0`, served model
  `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`; whole run ~12 min — bootstrap ready ~+700s, infer ~30s — on
  a 24 GB L4. **Requires `colab` ≥ 0.6.0** (keep-alive prerequisite above): the ~12-min cold start only
  finishes once the VM stops idle-pruning.) NVFP4 is Blackwell-native
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

## 2026-06-18 — Live web search for deep research WORKS (Ollama backend)

- **OpenClaw now executes REAL web search** (`web_search`/`web_fetch` → Brave) on a fee-free T4. Run:
  `--config configs/lfm2_ollama_web.json --task examples/web_verify_task.json`. Validated: the agent
  did multi-step search→fetch, returned cited python.org URLs + the live version, and answered by name
  ("Your name is Hiroki").
- **Why it was broken:** `python -m llama_cpp.server` (llama-cpp-python) has NO tool parser for LFM2.5's
  Pythonic `<|tool_call_start|>[...]<|tool_call_end|>` calls — it returns them as plain TEXT, so OpenClaw
  never executes them. Native `llama-server --jinja` parses them but needs llama.cpp PR #24178
  (2026-06-05) and there is NO prebuilt Linux-CUDA binary that recent (oobabooga's newest is pre-fix;
  ggml-org ships Linux cpu/vulkan/sycl/rocm but CUDA only for Windows). **Fix = serve via OLLAMA**
  (`serve.backend: "ollama"`): prebuilt CUDA (no compile), current llama.cpp, its own template parser →
  OpenAI `/v1` returns STRUCTURED `tool_calls`.
- **`ollama` backend** (`install_ollama`/`start_ollama`): `apt-get install zstd` FIRST (Colab lacks it;
  use `-o DPkg::Lock::Timeout=300` — the OpenClaw bg-installer holds the apt lock) → `ollama.com/install.sh`
  → `OLLAMA_HOST=127.0.0.1:8000 OLLAMA_CONTEXT_LENGTH=<n> ollama serve` → `ollama pull lfm2.5:8b`. Model
  id is an Ollama tag (`lfm2.5:8b` = LFM2.5-8B-A1B, "tools" capability). `compat.supportsTools:true`.
  Raise `num_ctx`/`contextWindow` (65536) — OpenClaw's prompt budget is contextWindow/2 and multi-step
  tool results accumulate in the shared session (overflowed at 32768).
- **Web/identity wiring is config-gated** (`_configure_web_and_identity`): `openclaw.web` installs the
  EXTERNAL brave plugin (`openclaw plugins install @openclaw/brave-plugin`), trusts it (`plugins.allow`),
  enables `tools.web.*`, sets `tools.profile coding`; `openclaw.identity.name` seeds workspace `USER.md`
  (injected every agent session — the "remember my name" fix). `lean_workspace` trims the 8 KB default
  AGENTS.md so a small model doesn't overflow.
- **Secrets:** `BRAVE_API_KEY` is forwarded from the controller's `~/.env` via a strict ALLOWLIST
  (launcher → `/content/ocdg_secrets.json` → `oc_env`), NEVER the user's `OPENCLAW_GATEWAY_TOKEN`. **T4
  default is now LFM2.5** (`llama_lfm2.json`); Qwen3.5-9B (hybrid-SSM) crashes llama.cpp mid-generation
  on a T4 — don't use it there.
- **DiffusionGemma/L4 ALSO has web search** (`configs/diffusiongemma_web.json`, VERIFIED on L4 2026-06-18):
  vLLM NATIVE tool_calls via the `gemma4` parser — append `--enable-auto-tool-choice --tool-call-parser
  gemma4 --reasoning-parser gemma4` to the vLLM serve_args (per the official recipes.vllm.ai DiffusionGemma
  recipe; NO `--chat-template` — built-in template handles tools; thinking stays ON, routed to
  reasoning_content), set `compat.supportsTools:true`, and raise `--max-model-len`/`contextWindow` to 32768
  (the plain 4096 overflows the agent+tools prompt even on "what is my name?"; KV ~0.7 GiB easily fits the
  L4). Same `openclaw.web`/`identity` wiring — NO code change. Verified: gemma4 emitted native tool_calls
  under block-diffusion decode, web_search hit Brave (Python 3.14 + cited URL), "Your name is Hiroki",
  finishReason stop, no tag leakage. Multi-step edge: a 3rd accumulated tool step hit OpenClaw's "Already
  compacted" auto-compaction bug (also seen on T4) — raise context further (65536) for heavier multi-step.

## 2026-06-22 — Bounded-context deep research (Layers 1–2) + Layer-3 subagent FAN-OUT, VERIFIED on L4

The "Already compacted" multi-step edge above is now **SOLVED** with OpenClaw's own bounded-context machinery — **NOT** "raise the window" (the weakest lever; the docs say LOWER the reserve).

- **Layers 1–2 (config-only, VERIFIED on T4, commit `6142120`):** a gated `openclaw.context` block (`_configure_context`) turns ON `contextPruning` (**OFF by default for non-Anthropic backends** — the key lever; it trims old multi-KB tool results between calls), LOWERS `compaction.reserveTokensFloor`→0 / `reserveTokens`→4096 (the "contextWindow/2 budget" is the floor eaten from a small window, not a hard /2), enables `midTurnPrecheck`, caps `contextLimits.toolResultMaxChars`, sets `memorySearch.provider`. Proof: `configs/lfm2_ollama_research.json` + `examples/web_research_deep.json` (6 steps / 4 accumulating searches) ran CLEANLY at contextWindow 32768 — the exact size that hit "Already compacted" at step 3 without it (pruning fired: `toolResultReducibleChars` 0→15k).
- **Layer 3 (subagent fan-out, VERIFIED on L4/DiffusionGemma 2026-06-22, commit `b52be9b`):** set the task's `orchestration: "subagent-fanout"` (default stays `shared-session`). One LEAD turn delegates each sub-question to an ISOLATED child via `sessions_spawn(context:"isolated")` + `sessions_yield`; raw web pages stay in the child transcript, only a distilled summary returns → the lead stays bounded regardless of page count. Run `--gpu L4 --config configs/diffusiongemma_research.json --task examples/web_research_fanout.json`. Proof: the lead spawned 2 ISOLATED children, each ran real Brave `web_search`/`web_fetch` (raw 96–140 KB pages quarantined in the children), and the lead synthesized a cited Markdown table (Python / Node LTS) in **~47 s** with **`compactionCount 0`**. Confirmed green twice (`manifest.ok:true`, table in `research_result.md`).
- **Two HARNESS gotchas fixed in `b52be9b` (NOT architecture):** (1) `run()`'s `subprocess.run(text=True)` returns `TimeoutExpired.output` as **bytes** → decode it or the timeout path raises `TypeError("can't concat str to bytes")` and loses the whole phase. (2) `openclaw agent --local --json` **HANGS ~20 min after producing its answer once subagents are spawned** (doesn't self-exit until children are reaped) → it gets killed by timeout and never prints `--json`. Recover the synthesis from the live server-side trajectory (`_lead_synthesis_from_trajectory`: last `model.completed.assistantTexts` for sessionKey `agent:main:<session_key>`); the fan-out success check keys on `got_text`, **NOT** the CLI returncode (124 is expected). Keep the lead timeout SHORT (the answer is fast; you're only waiting to kill a hung process). See memory `openclaw-local-subagent-cli-hang`.
- **Practical guidance:** for T4 fee-free, **Layer-1 pruning** is the bounded-context fix; **Layer-3 fan-out is the L4 path** (LFM2.5-8B on a serial T4 spawned + searched correctly but was too slow to finish the orchestration in-budget — needs a capable model on vLLM).
