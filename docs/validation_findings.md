# Validation findings (live log)

Status of bringing up **OpenClaw → vLLM → (target) DiffusionGemma** on a Google Colab GPU,
controlled from local Claude Code via the `colab` CLI. This file records what has actually
been **run and proven** on Colab, as opposed to the aspirational design in `architecture.md`.

Last updated: 2026-06-22. Account: free-tier consumer Colab (`hafnium49@gmail.com`).

## TL;DR

- **✅ FULL END-TO-END GREEN achieved (run #6, 2026-06-15)** on a free Colab T4 with a small
  model (`Qwen/Qwen2.5-0.5B-Instruct`): `openclaw infer model run --gateway` returned
  `{"ok": true, "transport": "gateway", "provider": "vllm", "outputs":[{"text":"openclaw-vllm-ok"}]}`
  (`infer_rc=0`, `infer_ok=true`). Whole pipeline — provision → install → serve → onboard →
  gateway → infer → download → teardown — runs in ~11 min, no VM reclaim, no websocket drop,
  no leaked session.
- **The full green required two OpenClaw fixes** beyond onboard (a content-format compat flag
  and a token-budget alignment) **plus** the decoupled short-exec architecture (details below).
- **The orchestration had to be re-architected** from "one long `colab exec`" to a
  **decoupled short-exec** design, because a single streaming exec drops (`Connection was
  lost`) around ~10.5–11 min and the vLLM cold start alone is ~7 min.
- **Blocker for the real target:** `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` needs ~24 GB
  VRAM (L4+). **L4 is not entitled on this free account** (`Backend rejected accelerator
  'L4'`). The proven small-model path lifts to DiffusionGemma only on a bigger GPU
  (Colab Pro/Enterprise or a rented L4/A100).
  **(Superseded 2026-06-17: L4 provisions fine now — the real blocker was the keep-alive bug
  below, not the GPU.)**

## ⚠️ The "~10-minute lifetime" was a colab-cli keep-alive bug (root cause + fix, 2026-06-17)

The single biggest source of lost runs — including "DiffusionGemma reaches serve but never
completes" — was **not** our workload. `google-colab-cli ≤ 0.5.x` keeps a VM alive via
`RuntimeService/KeepAliveAssignment` with a **hardcoded quota project `1014160490159`**; for ordinary
external accounts that RPC returns **`403 USER_PROJECT_DENIED`**, so Colab **idle-prunes the runtime at
~10–12 min**. Critically this cap is **independent of kernel activity** — proven by a continuous
heartbeat exec (kernel busy every 5 s) that still died at ~12 min, on **both T4 and L4**. Any bootstrap
longer than ~10 min (DiffusionGemma's vLLM-nightly install + ~13 GB NVFP4 download + load + warmup ≈
20–40 min) therefore cannot finish.

**Fix: upgrade the CLI to ≥ 0.6.0** (`uv tool upgrade google-colab-cli`, or `colab update --install`).
0.6.0 (2026-06-15) replaced the project-scoped RPC with a **tunnel-frontend keep-alive ping**
(`GET https://colab.research.google.com/tun/m/<endpoint>/keep-alive/`) that needs no project quota and
works for everyone. After upgrading, `~/.config/colab-cli/colab.log` shows the `tun/.../keep-alive/`
GETs and **no `USER_PROJECT_DENIED`**, and the VM survives past ~12 min — so the long DiffusionGemma
cold start can complete. **Confirmed 2026-06-17:** the full `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`
run went GREEN end-to-end on an L4 (gateway → vLLM → infer, `ok:true` / `got_text:true`, real
thinking-mode reply, ~12 min, clean teardown) — the project's actual target, finally reached.

Also hardened on `main` (2026-06-17): the launcher's `poll_worker` wraps the status upload/exec in
`timeout` via a resolved `$COLAB_BIN`, so a flaky kernel websocket that hangs one exec for minutes
can't stall the poll loop past the prune. (`timeout` execs a real binary; it can't call the `colab`
shell function, and `timeout command colab …` fails because `command` is a builtin.)

## Validated stack (T4)

| Component | Detail |
|---|---|
| GPU | Colab T4, 15 GB VRAM, ~66 GB disk, Turing (sm_75, no FlashAttention-2 → FlashInfer) |
| vLLM | `0.23.0`, needs CUDA 13 (`libcudart.so.13`) |
| torch | Colab preinstalls `2.11.0+cu128`; must be replaced with `+cu130` |
| OpenClaw | `2026.6.6 (8c802aa)`, npm-based install |
| Small model | `Qwen/Qwen2.5-0.5B-Instruct` (smoke model in place of DiffusionGemma) |

## Colab CLI / auth

- Install with `uv tool install google-colab-cli` (not pip into a conda base).
- **Use `--auth=adc`.** The default `oauth2` hangs (no client config). ADC must include the
  `colaboratory` scope: `gcloud auth application-default login --scopes=...,https://www.googleapis.com/auth/colaboratory`.
- **Isolate session state with `--config <file>`** so concurrent/other invocations don't
  prune the live session. A stray `colab status`/`sessions` against the default state file
  killed a live run early.
- **Keep-alive: upgrade `colab` to ≥ 0.6.0.** Older CLIs' `KeepAliveAssignment` RPC returns 403
  `USER_PROJECT_DENIED`, so the VM is idle-pruned ~10–12 min **even with the kernel kept busy** (NOT
  moot — proven by a continuous heartbeat). 0.6.0 uses a tunnel keep-alive that works. See "The
  keep-alive bug" section above.
- `colab exec` cannot pass args and keeps no state between calls — pass inputs via uploaded
  files; default exec timeout is short, so pass `--timeout`.

## The architecture correction (most important finding)

Originally the plan was "run the whole bootstrap as ONE long chatty `colab exec`."
**Five end-to-end runs proved this is wrong for jobs > ~10 min:**

| Run | Single-exec wall time | Result |
|---|---|---|
| vtest (vLLM only) | ~9 min | ✅ survived, downloaded |
| #2 | ~11 min (infer landed just in time) | ✅ downloaded |
| #3 | ~10.5 min | ✅ downloaded |
| #4 | ~10.8 min | ❌ `RuntimeError: Connection was lost` at ~10.6 min; no result |

A single **streaming** exec's websocket (jupyter-kernel-client) gets flaky around
~10.5–11 min and drops — **even with a keepalive thread printing every 5 s** (the keepalive
prevents *idle* drops but not this lifetime ceiling). Since vLLM install (~2.5 min) +
cold-start warmup (~7 min) ≈ 9.5 min must elapse before the inference, a single exec has
almost no margin.

**Correct architecture = decouple into short execs, none waiting through the warmup:**

1. **boot** (~3 min): install vLLM + OpenClaw, `openclaw onboard`, apply config, then launch
   `vllm serve` **detached** (`nohup … &`) and exit. The 7-min warmup now happens *after*
   the exec returns.
2. **poll** (~5 s each, every ~30 s): short execs that check `127.0.0.1:8000/v1/models`.
   Dual purpose — detect when warmup finishes **and** keep the kernel active so Colab's
   idle timer never reaches the ~10-min reclaim.
3. **finish** (~30 s): start a fresh gateway against the now-ready vLLM, run the inference,
   write the result JSON.

No single exec exceeds ~3 min → comfortably under the flaky zone. The earlier detached
design failed only because it polled too sparsely (kernel idle too long); **frequent** short
polls fix that. The VM stays alive because the kernel is touched every ~30 s and `vllm serve`
holds the GPU.

> Note: the committed `bin/colab_openclaw_diffusiongemma.sh` + `remote/…py` still use the
> older detached-bootstrap + sparse-poll design and need refactoring to this short-exec
> model. The **proven** path today is the dev harness under `runs/dev/` (below).

## vLLM on Colab T4

- **cu13 install fix:** vLLM 0.23 needs CUDA 13, but Colab ships torch `+cu128`. Recipe:
  ```bash
  python -m pip install -U uv
  python -m pip uninstall -y torch torchvision torchaudio || true
  uv pip install --system --torch-backend auto vllm     # pulls torch 2.11.0+cu130
  ```
  Serve with `LD_LIBRARY_PATH` pointing at the nvidia pip libs
  (`/usr/local/lib/python*/dist-packages/nvidia/*/lib`).
- **Cold start ~7 min** on T4 (model load ~15 s + memory profiling + FlashInfer attention
  warmup ~2.5 min). It does **not** hang — `--enforce-eager` for stability. Ready at
  ~408–448 s; `/v1/models` returns 200.
- **Direct probe is perfect:** a raw `/v1/chat/completions` call returns the exact expected
  text with `finish_reason=stop`. vLLM and the model are not the problem in any OpenClaw
  failure (this was the decisive diagnostic).

## OpenClaw wiring + the two infer fixes

`openclaw onboard --non-interactive --accept-risk` with `--custom-provider-id vllm` /
`--custom-base-url …/v1` / `--custom-model-id …` / `--custom-compatibility openai` /
`--custom-api-key "$VLLM_API_KEY"` plus loopback token gateway flags wires the provider in
one call. `--accept-risk` is **required** with `--non-interactive`. Run the CLI with the npm
global bin on PATH: `export PATH="$(npm prefix -g)/bin:$PATH"`. Start the gateway manually
(`nohup openclaw gateway run`) since the Colab container has no systemd (`--skip-daemon`).

Onboard alone is **necessary but not sufficient** — the gateway infer initially failed with
`GatewayClientRequestError: FailoverError: … incomplete terminal response: code=incomplete_result`
(gateway log: `stopReason=stop payloads=0` = empty completion). Two fixes, applied after
onboard and before starting the gateway, via `openclaw config set` (only the **`[0]`** index
form is valid — `models[]` errors with `Invalid path (empty "[]")`):

1. **Content format** — `models.providers.vllm.models[0].compat.requiresStringContent true`
   (+ `compat.supportsTools false`). OpenClaw sends `content` as a structured array; the
   local OpenAI-compat backend wants a plain string. Official runbook remedy.
2. **Token-budget overflow** — the gateway log then showed `reason=overflow`. Root cause:
   the model's `maxTokens` (4096) equalled vLLM `--max-model-len` (4096), so output budget +
   prompt exceeded the window → vLLM 400 → empty turn. Fix: serve with
   `--max-model-len 8192` **and** cap `models.providers.vllm.models[0].maxTokens 1024`
   (+ `contextWindow 8192`).

## Dev harness (proven path)

Under `runs/dev/` (gitignored scratch, kept on disk):

- `e2e.sh` — decoupled runner: `new` T4 → `boot` → `poll` loop → `finish` → download → teardown.
- `e2e_boot.py` / `e2e_poll.py` / `e2e_finish.py` — the three exec phases.
- `e2e.py` — earlier single-exec version (documents the keepalive + auth-probe fixes; superseded by the split).
- `vtest.py` — vLLM-only proof (`vllm-t4-ok`).

Run: `bash runs/dev/e2e.sh` → success = `/tmp/e2e_result.json` has `"infer_ok": true`
(model returns `openclaw-vllm-ok`). Each phase also records `vllm_direct_*` (direct vLLM
probe) and `gateway_log_tail` for diagnosis.

## Lifting to DiffusionGemma (the actual target)

When an L4/A100 (24 GB+) is available, change in the harness/config:
`--gpu T4 → L4`, the model id to `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`, add the
diffusion serve flags from `configs/diffusiongemma_nvfp4.json`
(`--diffusion-config`, `--generation-config vllm`, `--hf-overrides`), set `HF_TOKEN` if the
checkpoint is gated. The OpenClaw compat/token-budget fixes above still apply (a capable
model may not need them, but they are harmless). DiffusionGemma's larger context/warmup makes
the decoupled short-exec architecture even more necessary.

## 2026-06-22 — Deep-research stack VERIFIED end-to-end (web search + bounded context + Layer-3 fan-out)

- **DiffusionGemma on L4 is no longer aspirational** — provisioned, served (vLLM, NVFP4 via Marlin
  FP4 weight-only), and driven end-to-end. The 2026-06-15 "L4 not entitled" blocker was the keep-alive
  bug (fixed by `colab` ≥ 0.6.0), not the GPU.
- **Live web search VERIFIED on both paths:** T4 fee-free via the `ollama` backend (LFM2.5 structured
  `tool_calls`) and L4 via vLLM's `gemma4` native tool parser — real Brave `web_search`/`web_fetch`,
  cited URLs, USER.md identity. (Detail in the agent/skill 2026-06-18 entries.)
- **Bounded-context Layers 1–2 VERIFIED on T4** (commit `6142120`): the gated `openclaw.context` block
  (pruning ON + lowered `reserveTokensFloor`/`reserveTokens` + `midTurnPrecheck` + `toolResultMaxChars`)
  ran a 6-step / 4-search task clean at contextWindow 32768 — the size that previously hit "Already
  compacted" at step 3. **"Raise the window" is the weakest lever; LOWER the reserve + turn pruning ON.**
- **Layer-3 subagent fan-out VERIFIED on L4/DiffusionGemma** (commit `b52be9b`, 2026-06-22): the task's
  `orchestration:"subagent-fanout"` makes a LEAD turn delegate each sub-question to an ISOLATED child
  (`sessions_spawn context:isolated` + `sessions_yield`); raw pages stay quarantined in the children.
  The lead synthesized a cited table (Python / Node LTS) in **~47 s** with **`compactionCount 0`**;
  confirmed green twice (`manifest.ok:true`, table in `research_result.md`). Two harness gotchas fixed in
  the same commit: decode `TimeoutExpired.output` (bytes→str, else `TypeError`), and recover the lead
  synthesis from the server-side trajectory because `openclaw agent --local --json` hangs ~20 min after
  answering once subagents are spawned. Full detail in `.claude/agents/colab-openclaw-diffusiongemma.md`.

## 2026-06-23 — Fan-out follow-ups live-verified; MULTI-LEVEL depth DISABLED (does not complete)

Four fan-out/context follow-ups were verified live (≈12 GPU runs). Three are solid; multi-level is
disabled.

- **✅ Early-exit on trajectory completion** — the fan-out lead is launched detached and the trajectory
  polled; the group is killed once a TERMINAL substantive synthesis exists AND the trajectory has been
  silent for `OCDG_EARLYEXIT_SILENCE_S` (180s). Verified: full cited table, no truncation, ~9× faster
  than the old blocking cap. (Bugs found+fixed en route: bytes-timeout crash; killing on an intermediate
  "still waiting…" turn; capturing a `NO_REPLY` memoryFlush turn; snapshot sorted by filename not mtime.)
- **✅ Parallel fan-out** — lead spawns all children before yielding; the 6-question fan-out completed.
- **✅ Layer-2 memory recall** — fixed: notes must be saved as `memory/*.md` (only those are FTS-indexed;
  the agent had written `memory/<name>` with no `.md`), `memorySearch.{provider:none, enabled:true}`,
  and the invalid `tools.memory.enabled` keys removed. Verified: `memory_search` returns FTS hits on the
  T4/Ollama/LFM2.5 path. (Recall completeness is limited by the 8B model reusing filenames — a model, not
  harness, limit.)
- **⚙️ Multi-level depth (LEAD → COORDINATOR → leaf) — STRUCTURE verified, COMPLETION not achievable →
  DISABLED by default.** A prescriptive coordinator-tier prompt (`_fanout_lead_message_multilevel`) +
  documented spawn caps (`agents.defaults.subagents.{maxSpawnDepth, maxChildrenPerAgent, maxConcurrent}`)
  make the tier FORM reliably (2 coordinators + 4 leaves, every run). But it **never completes end-to-end**
  on the available models, proven exhaustively:
  - `max-num-seqs 1` (DiffusionGemma-26B serial) → tree forms, times out (one coordinator always lags).
  - `max-num-seqs 4` (batch) → vLLM never starts (block-diffusion can't batch on a 24 GB L4).
  - smaller tree (7 agents), 40-min budget, 75→180s silence window → still ends on "waiting for
    coordinator-N" (DiffusionGemma too slow).
  - T4/Ollama/LFM2.5 + `OLLAMA_NUM_PARALLEL=4` (fast + concurrent) → the 8B model is too weak: it ECHOES
    the coordinator task strings as text instead of calling `sessions_spawn` → 0 tree.

  Root cause: **no available model is BOTH capable enough to orchestrate the nested tree AND fast enough to
  complete it in budget.** The code is retained but gated OFF behind `openclaw.fanout.multilevel` (default
  `false`); the default fan-out path is flat single-level (verified). Set `multilevel:true` to re-enable
  when a capable+fast model (or true batched concurrency on a capable model) is available.

## 2026-06-23 — Ported the wg-automation deep-research skill to OpenClaw (DiffusionGemma-optimized)

Converted wg-automation's `claude-deep-research-skill` (a Claude Code native, citation-backed research
engine) into the OpenClaw `deep-research` skill and tuned it for the DiffusionGemma/L4 path. The skill
now ships BOTH as a checked-in artifact `skills/deep-research/SKILL.md` AND as the `DEEP_RESEARCH_SKILL`
constant the remote writes to `~/.openclaw/skills/deep-research/SKILL.md`; **`self_test.py` AST-extracts
the constant and asserts it equals the artifact** so they can never silently diverge (the constant is
what actually reaches Colab). New config `configs/diffusiongemma_deepresearch.json` + task
`examples/web_research_citation.json`.

This was a **principled distillation, not a 1:1 port** — the source assumes Claude's huge context, ~10
Python helper scripts, `search-cli`/Exa, concurrent `Task` sub-agents, and HTML/PDF export, none of which
exist on the ephemeral loopback VM. What was KEPT vs DROPPED, and the harness-alignment facts:

- **Kept (the crown jewels):** the citation/anti-hallucination discipline — every fact cited `[N]` in its
  own sentence, FACT-vs-SYNTHESIS separation, a zero-tolerance complete bibliography, the **source-as-data
  trust boundary** (fetched pages are data to quote, never instructions — a prompt-injection guard), and
  realistic triangulation (≥2 independent sources per core claim, explicit single-source flags). Plus the
  8-phase method (scope→plan→retrieve→triangulate→outline→synthesize→critique→package) scaled down.
- **Persistence remap:** the `citation_manager.py`/`evidence_store.py` ledgers and `verify_citations.py`
  become an **append-only evidence store in `memory/ev-NN-<slug>.md`** (the note number NN *is* its `[N]`,
  so a citation cannot exist without a saved, recallable source), a stable `memory/_citations.md` N→URL map
  (drift backstop), and an **in-thinking citation self-check** before each section write (zero output-token
  cost; thinking is ON anyway). `source_evaluator.py` → in-note credibility banding.
- **DiffusionGemma optimizations:** per-turn output is hard-capped at `maxTokens=2048` (~1500 words), so the
  report is written **one section per turn** and the harness's existing per-step append to
  `research_result.md` IS the assembled report — `examples/web_research_citation.json` splits PACKAGE across
  5 section-sized turns so no turn overflows. Decode is serial (`--max-num-seqs 1`), so fan-out is used only
  for context isolation, never for speed; multi-level stays OFF. The skill is ONE lean file (~2k tokens) so
  it doesn't re-trip the bundled-skills overflow.
- **Dropped (with reason):** HTML/PDF/WeasyPrint + `~/Documents` (headless VM, no display); `search-cli`/Exa
  (only the Brave plugin exists); the Python scripts + `*.jsonl` ledgers (not installed; replaced by
  `memory/*.md`); 20k-word single-shot/ultradeep reports (impossible at 2048 tok/turn); a self-managed
  report file (the harness OWNS `research_result.md` and captures the turn TEXT — a separate file would be
  invisible, so the skill reserves `write` for memory notes only).
- **Harness-alignment facts (so they aren't re-litigated):** (1) the agent's TURN TEXT is the captured
  artifact, appended under `## Step i` (shared-session) / `## Lead synthesis` (fan-out) — the skill must not
  keep its own report file. (2) In `subagent-fanout` the harness injects its OWN spawn-all-then-yield lead
  prompt + per-child caps; the skill defers to that choreography (it does not issue its own
  `sessions_spawn`) and instead makes per-child research + lead synthesis/citation excellent. Fan-out needs
  a SEPARATE bare-subquestion task (`examples/web_research_fanout.json`), NOT flipping `orchestration` on the
  citation task's phase-verb steps. The evidence ledger is snapshotted to `openclaw_state/memory/` for audit.

Design was produced via a 4-design judge panel → synthesis → adversarial review; the review's fixes (split
PACKAGE so no step exceeds one section; same-turn-vs-earlier-turn memory recall timing; fan-out LEAD
reworded from a "do NOT spawn" prohibition to a deference framing; bounded per-turn thinking) are folded in.

**VERIFIED end-to-end on L4/DiffusionGemma 2026-06-23 (`runs/deepresearch3`, `manifest.ok:true`).** It took
three live runs, each surfacing a real fix:
- Run 1 — the agent replied "no topic" every step: `task.topic` is header-only and never reaches the prompt
  (convention is self-contained steps). Fix: state the topic verbatim in step 1.
- Run 2 — step 1 produced a textbook scope, but steps 2-10 hit "Context overflow". TWO environmental causes:
  (a) **Brave free plan 429-rate-limits BURSTS** — the gemma4 model fired parallel `web_search` calls, all
  429'd, and the retry storm ballooned the turn (prePromptMessageCount 4→37) until overflow; (b) OpenClaw
  reserves **contextWindow/2** for output regardless of `reserveTokens`, so 32768 left only ~16k usable
  prompt. (A single date `web_search` succeeded earlier the same day, proving it's the per-second burst
  limit, not monthly quota.) Fixes: a skill **ONE-SEARCH-AT-A-TIME** rule (no bursts; on 429 note the gap and
  proceed, never storm-retry) and **--max-model-len/contextWindow 32768→49152** (~24576 usable; fits the L4's
  measured **53,394-token KV cache** at 1.08x — 65536 would not).
- Run 3 — SUCCESS: 22 `web_search` calls succeeded (429s 20→4, overflow 7→4 and auto-recovered), 5 sources
  saved as `memory/ev-01..ev-05` + `_citations.md`, a triangulation table (CONFIRMED ≥2 independent
  publishers), a red-team critique, and a progressively-built cited report — exec summary → findings
  (consensus-vs-debate) → labelled SYNTHESIS → Limitations (correctly **flagging the single-source
  $0.0003 metric [4]** and a 403 fetch gap) → Recommendations → a COMPLETE References `[1]-[5]` with real
  URLs (no placeholders). Audit: every body `[N]` resolves to a saved ev-note; `_citations.md` matches the
  bibliography; **no fabricated citations**.

Operational notes for future runs: (1) the Brave FREE plan only sustains ~1 search/sec — the
one-search-at-a-time rule is load-bearing, not advisory; a burstier task will 429 and overflow. (2) The
contextWindow/2 output reserve means usable prompt budget is HALF the window — size the window for the
accumulating shared session, not just the single largest turn. (3) DiffusionGemma occasionally emits a stray
unicode glyph (e.g. `독`) at a section break — a decoder artifact, cosmetic, not a pipeline bug.

## Open items

- [x] Land `infer_ok=true` on T4 via the decoupled harness — **done, run #6, 2026-06-15.**
- [x] Live-verify the ported `deep-research` skill end-to-end on L4 (`configs/diffusiongemma_deepresearch.json`
      + `examples/web_research_citation.json`) — **done 2026-06-23, run 3 (`runs/deepresearch3`, `manifest.ok:true`):**
      complete cited `research_result.md`, every body `[N]` resolves to an `openclaw_state/memory/ev-0N-*.md` note,
      `_citations.md` matches the bibliography, single-source claim flagged, no fabricated citations. Took the
      anti-burst + window-49152 fixes above to land.
- [ ] Refactor `bin/` + `remote/` from detached-bootstrap+sparse-poll to the short-exec model
      (port `e2e_boot.py`/`e2e_poll.py`/`e2e_finish.py` into the launcher; update `self_test.py`).
- [x] Obtain an L4/A100 and run the real DiffusionGemma profile — **done: L4 e2e green 2026-06-17;
      web search + bounded-context + Layer-3 fan-out all VERIFIED on L4 by 2026-06-22 (see above).**
