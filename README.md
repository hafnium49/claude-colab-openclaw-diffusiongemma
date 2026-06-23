# Claude Code · Colab · OpenClaw + self-hosted LLM appliance

Control a single Google Colab GPU runtime that hosts **OpenClaw** + a **self-hosted, OpenAI-compatible LLM** on loopback — no paid LLM API, no public tunnel. The serving backend is config-driven:

- **`llama_cpp` (default, validated, fee-free):** llama.cpp serves `LFM2.5-8B-A1B` (4-bit GGUF) on a free Colab **T4** (Qwen3.5-9B was the old default but its hybrid-SSM build crashes llama.cpp there). This is the path that actually works on free Colab — vLLM can't serve ≥3B on a T4 (FlashInfer crashes on Turing/sm_75; see [`docs/t4_llama_cpp_serving.md`](docs/t4_llama_cpp_serving.md)). For live web search add the `ollama` backend ([`configs/lfm2_ollama_web.json`](configs/lfm2_ollama_web.json)).
- **`vllm` (legacy):** vLLM serves `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` — needs an **L4** entitlement.

Everything runs on loopback inside the Colab VM (the sandbox); the local machine is only the controller. There are **two ways to deploy**:

| Mode | How | Use it for |
|---|---|---|
| **Headless / autonomous** | the bash master `bin/colab_openclaw_diffusiongemma.sh` (drives Colab via the `colab` CLI) | unattended runs, autonomous deep-research jobs, CI |
| **Manual / browser** | the **Colab notebook** `notebooks/openclaw_chat_colab.ipynb` (Run-all in your browser) | interactive testing, chatting, the inline OpenClaw dashboard |
| **Path A / free, no GPU** | the **Colab notebook** `notebooks/openclaw_colab_ai.ipynb` (browser, **CPU** runtime) | a fee-free chat with **no GPU and no API key**, when leaving the sandbox is acceptable |

The first two paths self-host the LLM **inside** the VM (fee-free *and* loopback-contained). **The bash master is the source of truth; the notebooks mirror it.** Path A is different: it wraps Colab's free `google.colab.ai` (Gemini), so it needs no GPU but runs inference on **Google's** backend — **not contained** (prompts leave the VM) and **browser-only** (the headless CLI can't fetch its proxy key).

---

## Deploy manually in the browser (Colab notebook)

This is the no-CLI path: open the notebook, Run all, chat. The open tab is the runtime heartbeat (keeps the VM alive); nothing is installed on your machine.

**1. Open the notebook in Colab:**

➡️ **https://colab.research.google.com/github/hafnium49/claude-colab-openclaw-diffusiongemma/blob/main/notebooks/openclaw_chat_colab.ipynb**

(Or, from the repo: `notebooks/openclaw_chat_colab.ipynb` → "Open in Colab".)

> **Private repo?** That link only works if the repo is **public**, *or* if you open it via
> Colab `File → Open notebook → GitHub`, sign in, and tick **"Include private repositories"**
> (a one-time GitHub authorization). An unauthenticated open of a private repo returns `404`.

**2. Select a GPU:** `Runtime → Change runtime type → T4 GPU → Save`.

**3. Run all:** `Runtime → Run all` (or run the cells top to bottom). **Keep the tab open.** The first run installs the llama.cpp wheel + OpenClaw and downloads the 5.6 GB GGUF — **~6 min, one time**.

**4. What each cell does** (mirrors the master's remote phases):

| Cell | Action | Master equivalent |
|---|---|---|
| 1 — Install | prebuilt `llama-cpp-python[server]` CUDA wheel (no compile) + OpenClaw | `bootstrap` → install |
| 2 — Serve | download the GGUF, serve llama.cpp on `127.0.0.1:8000`, wait until ready | `bootstrap` → `start_llama_cpp` |
| 3 — Onboard | onboard OpenClaw against `:8000` + the compat infer-fixes, start the gateway | `bootstrap` → `configure_openclaw` |
| 4 — 💬 Chat | one prompt via OpenClaw (edit `MESSAGE`, re-run per turn) | `prompt` → `_prompt_run` |
| 5 — 🤖 Autonomous task | a **multi-step** deep-research loop (`STEPS` → `/content/research_result.md`) | `task` → `_task_run` |
| 6 — Dashboard | the OpenClaw Control UI inline via Colab's port-proxy (no tunnel) | notebook-only |

**5. Chat:** edit `MESSAGE` in cell 4 and re-run. `Qwen3.5-9B` is a reasoning model — a turn takes ~1–2 min and shows a `<think>` trace; prefix `/no_think` for a fast, clean answer.

**6. Autonomous research:** edit `TOPIC` / `STEPS` in cell 5 and run — it answers each step with the self-hosted model (no API fee) and writes `research_result.md`. This is the in-cell equivalent of `bin/colab_openclaw_diffusiongemma.sh --task examples/research_task.json`.

> The notebook GUI/dashboard works only because *your* browser owns the runtime. A CLI-managed VM (the headless mode below) can't surface a browser dashboard — use the chat cell or the CLI there.

---

## Path A — free, no GPU (`google.colab.ai`)

For a **fee-free** chat with **no GPU, no model download, and no API key**, open
`notebooks/openclaw_colab_ai.ipynb` on a **CPU** runtime and Run all. It wraps Colab's free
`google.colab.ai` (Gemini) behind an OpenAI-compatible shim so OpenClaw can use it like any backend.

➡️ **https://colab.research.google.com/github/hafnium49/claude-colab-openclaw-diffusiongemma/blob/main/notebooks/openclaw_colab_ai.ipynb** (private-repo `404` caveat as above).

Two things to know — both intrinsic to `google.colab.ai`:

- **Not contained.** Inference runs on Google's servers, so your prompts **leave the VM**. The self-hosted paths above keep the model on the VM. Use Path A only when that's acceptable.
- **Browser-only.** `google.colab.ai` can only fetch its proxy key from the Colab **UI**, so this can't be driven by the `colab` CLI. The notebook serves the shim **in-kernel** (a background thread) and **primes the key in cell 1** so the shim can reuse it. The default model is `google/gemini-3.5-flash`; cell 1 prints the live `ai.list_models()` catalog and tells you to fall back to `google/gemini-2.5-flash` if it isn't offered yet.

---

## Deploy headless (the bash master)

Install the Google Colab CLI locally (`pip install google-colab-cli`), authenticate it (ADC), then:

```bash
# Cheap orchestration smoke (0.5B GGUF on T4 via llama.cpp) — validates the path fast
bash bin/colab_openclaw_diffusiongemma.sh --gpu T4 \
  --config configs/llama_smoke.json --task examples/prompt_task.json --out ./runs/smoke

# Validated default: LFM2.5-8B-A1B (4-bit GGUF) on a T4, single prompt (fee-free, self-hosted)
bash bin/colab_openclaw_diffusiongemma.sh --gpu T4 \
  --config configs/llama_lfm2.json --task examples/prompt_task.json --out ./runs/lfm2

# Deep research with LIVE web search (Ollama backend -> real Brave search) — needs BRAVE_API_KEY in ~/.env
bash bin/colab_openclaw_diffusiongemma.sh --gpu T4 \
  --config configs/lfm2_ollama_web.json --task examples/web_verify_task.json --out ./runs/research

# Original DiffusionGemma target on L4 (vLLM / NVFP4)
bash bin/colab_openclaw_diffusiongemma.sh --gpu L4 \
  --config configs/diffusiongemma_nvfp4.json --task examples/prompt_task.json --out ./runs/openclaw-dg

# Bounded-context deep research with Layer-3 subagent FAN-OUT (VERIFIED 2026-06-22 on L4) — needs BRAVE_API_KEY
# Lead delegates each sub-question to an ISOLATED sub-agent (sessions_spawn/yield); raw pages stay in the child.
bash bin/colab_openclaw_diffusiongemma.sh --gpu L4 \
  --config configs/diffusiongemma_research.json --task examples/web_research_fanout.json --out ./runs/fanout

# Citation-backed DEEP-RESEARCH REPORT (deep-research skill ported from wg-automation, tuned for DiffusionGemma)
# shared-session; research phases then PACKAGE split across 5 cited section-turns -> research_result.md. BRAVE_API_KEY in ~/.env.
bash bin/colab_openclaw_diffusiongemma.sh --gpu L4 \
  --config configs/diffusiongemma_deepresearch.json --task examples/web_research_citation.json --out ./runs/deepresearch
```

`--keep-session` leaves the VM up for inspection (default tears it down after download). The launcher runs `scripts/self_test.py` first, so a self-test failure aborts before any VM is provisioned. The `--out` directory receives `openclaw_diffusiongemma_results.zip`, `manifest.json`, `research_result.md` (research mode), `colab_session_log.ipynb`, and local command logs.

## Claude Code integration

The project ships a subagent and skill (`self_test.py` requires both):

```text
.claude/agents/colab-openclaw-diffusiongemma.md
.claude/skills/colab-openclaw-diffusiongemma/SKILL.md
```

Separately, the **OpenClaw deep-research skill** (the one the in-Colab agent runs) is checked in at `skills/deep-research/SKILL.md` and also shipped as the `DEEP_RESEARCH_SKILL` constant the remote installs into `~/.openclaw/skills/deep-research/` (`self_test.py` keeps the two byte-in-sync). It was ported from wg-automation's `claude-deep-research-skill` and optimized for the small DiffusionGemma context window: citation integrity (`[N]` per claim, complete bibliography, no fabrication), a `memory/ev-NN-*.md` evidence ledger recalled via `memory_search`/`memory_get`, and one-section-per-turn progressive report writing. Drop the file into any `~/.openclaw/skills/` to use it standalone.

Ask Claude Code to use the `colab-openclaw-diffusiongemma` subagent, or invoke the skill directly:

```text
/colab-openclaw-diffusiongemma run configs/llama_lfm2.json examples/prompt_task.json
```

## Notes

- **Fee-free + loopback containment.** The LLM is self-hosted (no paid API); the model + gateway are loopback-only inside Colab. Don't add public tunnels unless you accept the exposure.
- **Secrets via env only**, never in config files: `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` (model download) and `OPENCLAW_GATEWAY_TOKEN`/`VLLM_API_KEY` are read from the Colab environment if present.
- **Colab is ephemeral.** Treat every run as batch-style; nothing persists after the runtime ends. Free-tier T4 availability fluctuates (`colab new` can return `503` after heavy same-day use).
- Validate locally before any run: `python scripts/self_test.py` and `bash -n bin/colab_openclaw_diffusiongemma.sh`.
