---
name: colab-openclaw-diffusiongemma
description: Run a Colab CLI-controlled OpenClaw appliance where vLLM serves quantized DiffusionGemma inside the same Colab GPU runtime, then collect prompt outputs and logs.
allowed-tools: Bash Read Write Edit Glob Grep
---

# Colab OpenClaw DiffusionGemma skill

Use this skill for a job-oriented local Claude Code workflow that provisions a Colab GPU, starts vLLM, configures OpenClaw to use the Colab-local vLLM endpoint, runs a prompt through OpenClaw's inference CLI, and downloads a single result bundle.

## ⚠️ Prerequisite — `colab` CLI must be ≥ 0.6.0 (keep-alive bug)

Run `colab version` first. **≤ 0.5.x has a keep-alive bug**: its keep-alive RPC returns
`403 USER_PROJECT_DENIED` for external accounts, so Colab **idle-prunes the VM at ~10–12 min no matter
how busy the kernel is** (confirmed on T4 *and* L4, even under a continuous heartbeat) — silently
killing any bootstrap longer than ~10 min (vLLM / DiffusionGemma cold start). This is the root cause
behind "DiffusionGemma reaches serve but the run never completes". Fix:
`uv tool upgrade google-colab-cli` (or `colab update --install`). **0.6.0 (2026-06-15)** switched to a
tunnel keep-alive (`GET /tun/m/<endpoint>/keep-alive/`, no project quota) that works for everyone; the
VM then survives long enough for DiffusionGemma's full cold start. Verify: no `USER_PROJECT_DENIED` in
`~/.config/colab-cli/colab.log` and a session lives past ~12 min. (The launcher's `poll_worker` is also
`timeout`-hardened via `$COLAB_BIN` so a hung kernel exec can't stall the poll loop past the prune.)

## Standard workflow

1. Validate local files:

```bash
python scripts/self_test.py
```

2. Run a lightweight smoke test:

```bash
bash bin/colab_openclaw_diffusiongemma.sh \
  --session openclaw-dg-smoke \
  --gpu T4 \
  --config configs/smoke_test_tiny.json \
  --task examples/prompt_task.json \
  --out ./runs/smoke
```

3. Run the full quantized DiffusionGemma profile:

```bash
bash bin/colab_openclaw_diffusiongemma.sh \
  --session openclaw-dg \
  --gpu L4 \
  --config configs/diffusiongemma_nvfp4.json \
  --task examples/prompt_task.json \
  --out ./runs/openclaw-dg
```

4. Inspect:

```bash
unzip -l ./runs/openclaw-dg/openclaw_diffusiongemma_results.zip
cat ./runs/openclaw-dg/manifest.json 2>/dev/null || true
```

## Inputs

- Config JSON: model, vLLM serve flags, OpenClaw settings, ports, install policy.
- Task JSON: prompt, preferred transport, output file names.

## Outputs

- `openclaw_diffusiongemma_results.zip`
- `manifest.json`
- `openclaw_infer.json` or `openclaw_infer.txt`
- `vllm_models.json`
- `environment.txt`
- `vllm.log`
- `openclaw_gateway.log`
- `colab_session_log.ipynb`

## Safety and reliability

- Keep the Gateway on loopback unless the user explicitly accepts tunnel risk.
- Avoid mounting Drive unless the user explicitly needs persistent storage.
- Pass secrets via environment variables or Colab-side secrets, never by writing them into config files.
- Use `--keep-session` only when the user wants to inspect the running appliance.
- If the full DiffusionGemma checkpoint does not fit the GPU, report the exact failing step and keep the result zip.

## Validated path and gotchas (2026-06-15)

The dev harness (`runs/dev/e2e.sh`) is **confirmed green** end-to-end on a Colab T4 (run #6,
2026-06-15: `infer_ok=true`, model returns `openclaw-vllm-ok`). The standard `bin/` master has
since been **refactored (2026-06-17)** to this proven short-exec model: config-driven serve
backend (llama.cpp/LFM2.5-8B-A1B default — Qwen3.5-9B crashes llama.cpp on a T4 — vLLM/ollama also), all heavy phases detached + polled, compat
infer-fixes applied, and an autonomous `mode:"research"` task phase. Full details in
`docs/validation_findings.md`.
Key points:

- **Run the proven harness:** `bash runs/dev/e2e.sh` provisions a T4 and runs the validated
  decoupled flow (`e2e_boot.py` → `e2e_poll.py` ×N → `e2e_finish.py`). Success =
  `/tmp/e2e_result.json` contains `"infer_ok": true`.
- **Auth:** the CLI must be invoked `colab --auth=adc --config <state-file> …` (default
  `oauth2` hangs; ADC needs the `colaboratory` scope). Don't run a second concurrent `colab`
  command on the same state file during a run.
- **No single long exec:** a streaming `colab exec` drops around ~10.5–11 min and the vLLM
  cold start is ~7 min, so the work is split into short execs (boot/poll/finish), none of
  which waits through the warmup. Frequent short polls keep the VM alive.
- **vLLM cu13 fix:** drop preinstalled `torch+cu128`, then
  `uv pip install --system --torch-backend auto vllm`; serve with `--enforce-eager` and
  `LD_LIBRARY_PATH` set to the nvidia pip libs.
- **OpenClaw infer fixes (after onboard, via `openclaw config set …models[0]…`):**
  `compat.requiresStringContent true` (+ `compat.supportsTools false`) and a token budget with
  model `maxTokens` < vLLM `--max-model-len` (serve `8192`, `maxTokens 1024`). Otherwise the
  gateway returns `incomplete_result`.
- **GPU reality (updated 2026-06-17):** the account now has **Colab Pro + compute units**, so
  **L4/A100 work** (the old "L4 not entitled" no longer applies). DiffusionGemma-26B runs on an L4;
  prefer L4 over A100 (~3× cheaper). T4 still serves the llama.cpp models. See the 2026-06-17 section.

## llama.cpp / Qwen3.5-9B + notebook counterpart (2026-06-16)

The vLLM green above was only **0.5B**. **vLLM can't serve ≥3B on a T4** (Turing/sm_75 +
FlashInfer `BatchPrefillWithPagedKVCache` crash), so the **Qwen3.5-9B** floor model runs via
**llama.cpp** — confirmed green (OpenClaw → llama.cpp → 9B, `infer_ok=true`, ~35 tok/s). Full
recipe in `docs/t4_llama_cpp_serving.md`.

- **Master harness:** `runs/dev/relaunch.sh` (→ `llama_boot.py`/`llama_poll.py`/`llama_finish.py`)
  provisions a T4, serves the GGUF on `127.0.0.1:8000` via the **prebuilt** `llama-cpp-python[server]`
  cu124 wheel (no on-VM compile), onboards OpenClaw, verifies infer, and **leaves the session up**.
- **Port `:8000`, not `:8080`** (Colab's `node` owns 8080). **openclaw via absolute path**
  (`shutil.which("openclaw") or "/usr/bin/openclaw"`). **Chat = direct infer** (no `--gateway`):
  `runs/dev/chat.py "your message"`.
- **No browser attach to a CLI VM:** Colab hands a fresh CPU runtime instead (notebook-hash bind
  + ignored `dbu` dev flag). The `:18789` dashboard needs a tunnel **or** a browser-owned runtime
  (`serve_kernel_port_as_iframe`). T4 can return `503` (GPU cooldown) after heavy same-day use.
- **Notebook counterpart:** `notebooks/openclaw_chat_colab.ipynb` (built by
  `notebooks/_gen_notebook.py`) mirrors the harness phases as Run-all cells (install → serve →
  onboard → chat → autonomous task → inline dashboard). **The bash harness is master; mirror the
  notebook to it, not the reverse.** Roadmap: fee-free self-hosted LLM running autonomous,
  human-free jobs (deep research) — cell 5 scaffolds it.

## 2026-06-17 — LFM2.5, native-agent research, DiffusionGemma/L4, cost + keep-session

- **Configs now validated GREEN on T4 (llama.cpp):** `llama_qwen9b.json` (Qwen3.5-9B, best quality)
  and **`llama_lfm2.json`** (LiquidAI **LFM2.5-8B-A1B**, ~134 tok/s ≈ 4× faster; needs the prebuilt
  cu124 wheel **0.3.30** for the `lfm2moe` arch).
- **`mode:"research"` now drives the NATIVE OpenClaw agent**, not a hand-rolled loop:
  `openclaw agent --local --agent main --session-key <shared> --message <step> --json` per step
  (`--local` avoids the gateway operator-scope issue; shared `--session-key` = context across steps).
  Onboard **without `--skip-skills`**, install a `deep-research` SKILL.md in `~/.openclaw/skills/`, and
  **scope to it** with `openclaw config set agents.defaults.skills '["deep-research"]'` — otherwise the
  ~20 bundled skills overflow a small model's prompt (~8.9k tokens). Validated GREEN (step 4 synthesizes).
- **DiffusionGemma on L4 (Path B) — FULL END-TO-END GREEN, 2026-06-17** (gateway → vLLM → 26B-NVFP4
  returned a real thinking-mode completion: `ok:true`, `got_text:true`, served model
  `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`; ~12 min total on a 24 GB L4. **Needs `colab` ≥ 0.6.0** — the
  long cold start only completes once the VM stops idle-pruning; see the prerequisite up top.) NVFP4
  loads on L4 via vLLM's Marlin FP4
  **weight-only** fallback (no Blackwell needed); `DiffusionGemmaForBlockDiffusion` loads via
  `--trust-remote-code`. Two required fixes now in the harness/config: `start_vllm` **shlex-quotes each
  serve arg** (JSON args were shell-stripped → "invalid loads value"), and the config passes
  **`--max-model-len 8192`** (else vLLM reserves KV for the 256K context → OOM). Use RedHat's recipe.
- **Cost (`[[colab-gpu-costs]]`):** L4 ≈ ¥57/hr, A100 ≈ ¥170/hr (~3× L4), T4 cheapest; a DiffusionGemma
  L4 bootstrap ≈ ¥45 (~2.5 compute units; the cold start is ~20 of the ~32 min). **Reuse a warm session
  to skip it:** run 1 `--keep-session` (leaves the L4 warm), later runs add `--reuse-session` (same
  `--session NAME`) → the launcher ATTACHES by name (no `colab new`) and SKIPS bootstrap, running only
  the task (~10 min, ~0.8 units). Handle persists in `./runs/.sessions/<name>.json`; omit `--keep-session`
  on the last run to tear down. Historical gotcha (now SOLVED by `--reuse-session`): plain `--keep-session`
  + a fresh launcher invocation used to `colab new` a SECOND same-named runtime (duplicate billing) — never
  do that; use `--reuse-session` for warm re-runs. **Break-even (units = VM-up-time × rate, idle billed):**
  keeping the VM warm between runs costs units for nothing, so reuse only nets savings if you reuse within
  ~one cold-start of idle (~5 min T4, ~20 min L4) — else it goes net-negative; tear down on the last run.
  Verified on T4 2026-06-24 (cold 315 s → warm 49 s). Mechanism + compute-unit math:
  `docs/warm_session_reuse_and_costs.md`; the deep-research port: `docs/deep_research_port.md`. Kill
  orphaned/colliding sessions via the colab-cli client `unassign` API (`colab stop -s` can't reach
  store-less sessions). Always tear down promptly.

## 2026-06-18 — Live web search works (Ollama backend)

- **OpenClaw now executes REAL web search** (`web_search`/`web_fetch` → Brave) on a fee-free T4:
  `--config configs/lfm2_ollama_web.json --task examples/web_verify_task.json`. Validated: multi-step
  search→fetch, cited python.org URLs, and "Your name is Hiroki".
- **Why the llama.cpp path couldn't:** `python -m llama_cpp.server` has no parser for LFM2.5's Pythonic
  `<|tool_call_start|>[...]<|tool_call_end|>` tool calls → returns them as TEXT, never executed. Native
  `llama-server --jinja` needs llama.cpp PR #24178 (2026-06-05) and no prebuilt Linux-CUDA binary that
  recent exists. **Fix = `serve.backend: "ollama"`** — prebuilt CUDA, current llama.cpp, its own template
  parser → OpenAI `/v1` returns STRUCTURED `tool_calls`.
- **`ollama` backend:** install needs `apt-get install zstd` first (Colab lacks it; `-o DPkg::Lock::Timeout=300`
  because the OpenClaw bg-installer holds the apt lock), then `ollama.com/install.sh`,
  `OLLAMA_HOST=127.0.0.1:8000 OLLAMA_CONTEXT_LENGTH=<n> ollama serve`, `ollama pull lfm2.5:8b`. Model id =
  Ollama tag `lfm2.5:8b` (LFM2.5-8B-A1B, "tools"); `compat.supportsTools:true`; raise `num_ctx`/`contextWindow`
  to 65536 (budget = contextWindow/2; tool results accumulate in the shared session).
- **Config-gated wiring** (`_configure_web_and_identity`): `openclaw.web` installs the external brave plugin
  (`openclaw plugins install @openclaw/brave-plugin`), trusts it via `plugins.allow`, enables `tools.web.*`,
  `tools.profile coding`; `openclaw.identity.name` seeds workspace `USER.md` (per-session identity = the
  "remember my name" fix); `lean_workspace` trims the default 8 KB AGENTS.md. `BRAVE_API_KEY` is forwarded
  from `~/.env` via a strict ALLOWLIST (NEVER the user's `OPENCLAW_GATEWAY_TOKEN`). **T4 default is now
  LFM2.5** (`llama_lfm2.json`); Qwen3.5-9B (hybrid-SSM) crashes llama.cpp on a T4.
- **DiffusionGemma/L4 also has web search** (`configs/diffusiongemma_web.json`, VERIFIED on L4 2026-06-18):
  vLLM NATIVE tool_calls via the `gemma4` parser — add `--enable-auto-tool-choice --tool-call-parser gemma4
  --reasoning-parser gemma4` (official recipe; NO `--chat-template`; thinking ON via reasoning_content),
  `compat.supportsTools:true`, raise `--max-model-len`/`contextWindow` to 32768 (4096 overflows the
  agent+tools prompt). Same web/identity wiring (no code change). Verified: native tool_calls under
  block-diffusion, real Brave search (Python 3.14 + URL), "Your name is Hiroki". 3rd multi-step tool turn
  can hit OpenClaw's "Already compacted" bug — raise context (65536) for heavier multi-step.

## 2026-06-22 — Bounded-context deep research + Layer-3 subagent FAN-OUT (VERIFIED on L4)

The "Already compacted" multi-step edge above is **SOLVED** via OpenClaw's bounded-context machinery (not "raise the window").

- **Layers 1–2 (config-only, VERIFIED T4, `6142120`):** a gated `openclaw.context` block (`_configure_context`) turns ON `contextPruning` (OFF by default for non-Anthropic backends), LOWERS `compaction.reserveTokensFloor`→0 / `reserveTokens`→4096, enables `midTurnPrecheck`, caps `toolResultMaxChars`. `configs/lfm2_ollama_research.json` + `examples/web_research_deep.json` (6 steps / 4 accumulating searches) ran clean at contextWindow 32768.
- **Layer 3 (fan-out, VERIFIED L4/DiffusionGemma `b52be9b`):** set the task's `orchestration:"subagent-fanout"` (default `shared-session`) — one LEAD turn delegates each sub-question to an ISOLATED child (`sessions_spawn context:isolated` + `sessions_yield`); raw pages stay in the child, only a distilled summary returns. Run `--gpu L4 --config configs/diffusiongemma_research.json --task examples/web_research_fanout.json`. Proven: lead spawned 2 isolated children, each ran real Brave search, lead wrote a cited table (Python / Node LTS) in ~47 s, `compactionCount 0`; green twice (`manifest.ok:true`, table in `research_result.md`).
- **Two harness fixes (`b52be9b`):** (1) decode `TimeoutExpired.output` (bytes → else `TypeError("can't concat str to bytes")`); (2) `openclaw agent --local --json` hangs ~20 min after answering when subagents spawn → recover the synthesis from the server-side trajectory (`_lead_synthesis_from_trajectory`), judge on `got_text` not CLI rc (124 expected), keep the lead timeout short.
- T4 fee-free → **Layer-1 pruning**; **Layer-3 fan-out is the L4 path** (LFM2.5-8B on serial T4 was too slow to finish the orchestration).

## 2026-06-23 — Citation-backed deep-research REPORTS (ported deep-research skill)

The OpenClaw `deep-research` skill was upgraded by porting wg-automation's `claude-deep-research-skill`,
optimized for DiffusionGemma. It ships as `skills/deep-research/SKILL.md` and the `DEEP_RESEARCH_SKILL`
constant (kept in sync by `self_test.py`). Run a full cited report:

```bash
bash bin/colab_openclaw_diffusiongemma.sh \
  --gpu L4 \
  --config configs/diffusiongemma_deepresearch.json \
  --task examples/web_research_citation.json \
  --out ./runs/deepresearch
```

- shared-session: research phases build a `memory/ev-NN-*.md` evidence ledger; **PACKAGE is split across 5
  section-sized turns** (per-turn output is capped at `maxTokens=2048`) → a fully-cited `research_result.md`.
- Citation integrity is enforced in the skill (`[N]` per claim, complete bibliography, no fabrication,
  source-as-data trust boundary). The ev-note ledger is bundled under `openclaw_state/memory/` for audit.
- For fan-out RETRIEVE isolation use `examples/web_research_fanout.json` (bare sub-questions) — do NOT flip
  `orchestration` on the citation task. Needs `BRAVE_API_KEY` in `~/.env`.
- **VERIFIED end-to-end on L4 2026-06-23** (`runs/deepresearch3`): 5 sources → `ev-01..ev-05` + `_citations.md`,
  triangulation + critique, cited progressive report, complete `[1]-[5]` bibliography, no fabrication. Two
  run-tuned constraints: the Brave FREE plan only sustains ~1 search/sec (the skill enforces ONE-SEARCH-AT-A-TIME;
  bursts 429 → retry storm → context overflow), and OpenClaw reserves contextWindow/2 for output so the config
  raises the window to 49152 (~24576 usable; fits the L4's 53,394-token KV at 1.08x). See `docs/validation_findings.md`.
