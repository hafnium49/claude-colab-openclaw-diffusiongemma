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
backend (llama.cpp/Qwen3.5-9B default, vLLM legacy), all heavy phases detached + polled, compat
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
  L4 bootstrap ≈ ¥45. **`--keep-session` does NOT make a launcher re-run reuse the session** — it
  spins a SECOND same-named runtime (duplicate billing). Use it only for manual inspection; kill
  orphaned/colliding sessions via the colab-cli client `unassign` API (`colab stop -s` can't reach
  store-less sessions). Always tear down promptly.
