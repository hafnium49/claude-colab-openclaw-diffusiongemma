# Warm-session reuse (`--reuse-session`) and Colab compute-unit economics

Status: **`--reuse-session` VERIFIED live on T4 2026-06-24** (`runs/demo-t4-cold` → `runs/demo-t4-warm`).
Detail log: [`validation_findings.md`](validation_findings.md) (2026-06-24 sections).

## TL;DR

The DiffusionGemma cold start (model download + serve load) is ~20 of a run's ~32 minutes. `--keep-session`
alone could not be reused — a fresh launcher invocation calls `colab new`, spinning a *second* same-named
runtime (the documented duplicate-billing trap). **`--reuse-session`** attaches to an already-warm session
by name and skips both `colab new` and the bootstrap phase, so you pay the cold start **once** and each
subsequent run is just the task.

## The workflow

```bash
# run 1 — cold start (~32 min on L4 / ~5 min on T4), leave the GPU WARM:
bash bin/colab_openclaw_diffusiongemma.sh --gpu L4 --session warm-dg --keep-session \
  --config configs/diffusiongemma_deepresearch.json --task examples/web_research_citation.json --out ./runs/dr1

# run 2+ — WARM: skips colab new AND bootstrap, runs only the task:
bash bin/colab_openclaw_diffusiongemma.sh --gpu L4 --session warm-dg --reuse-session --keep-session \
  --config configs/diffusiongemma_deepresearch.json --task examples/web_research_citation.json --out ./runs/dr2

# last run — reuse then TEAR DOWN (omit --keep-session):
bash bin/colab_openclaw_diffusiongemma.sh --gpu L4 --session warm-dg --reuse-session \
  --config configs/diffusiongemma_deepresearch.json --task examples/web_research_citation.json --out ./runs/dr3
```

## How it works

- **Attach by name, no `colab new`.** Avoids the duplicate-runtime trap. A reuse run never provisions a VM.
- **Stable session-state file.** With `--keep-session`/`--reuse-session` the session handle persists in
  `./runs/.sessions/<name>.json` (keyed by `--session` NAME, not `--out`), so a reuse run with a *different*
  `--out` still attaches to the same warm VM. Plain one-shot runs keep the per-`--out` isolated state.
  Override with `OCDG_SESSION_DIR=` or `COLAB_CONFIG=`.
- **Aliveness gate.** `colab status` exits 0 even for a missing session (it just prints "... not found"), so
  the gate inspects the *output*, not the exit code. A dead/missing session errors and exits **before any
  upload — never provisioning a VM**.
- **Warm-backend health check.** A quick `bootstrap_status` exec confirms `bootstrap.done` persists on the
  warm VM before running the task; if the serve/gateway is gone it reports not-ready and collects
  diagnostics instead of running into a dead backend (drop `--reuse-session` to cold-start).
- **Teardown.** Respects `--keep-session` — omit it on the last run to stop the VM.
- **Only the TASK varies across reuse runs.** Serve/OpenClaw config changes need a fresh cold start; reuse
  re-uploads the remote + config + new task but does not restart the running serve/gateway.

## Compute-unit economics

Colab bills compute units = **(time the GPU VM is assigned) × (a fixed per-GPU rate)**. There is no separate
"unit" meter — burn is exactly proportional to wall time *while the VM is up, idle time included*. So the
cold-vs-warm unit ratio equals the wall-time ratio (rate-independent).

**Measured T4 demo** (one continuous billing: `colab new` → `colab stop`; T4 ≈ 1.96 units/hr):

| Phase | Duration | T4 units |
|---|---:|---:|
| 🥶 COLD START (provision + model download + load) | 286 s | **0.156** |
| run 1 task + bundle | 29 s | 0.016 |
| 😴 idle while kept warm (no work, still billed) | 50 s | 0.027 |
| 🔥 WARM RUN (reuse: attach + task + stop) | 49 s | **0.027** |
| **Total demo** | 414 s | **0.225** |

The **cold start burned ~5.8× the units of the warm run** (286 s vs 49 s) — ~70 % of the whole demo.

### The catch: idle billing and the break-even rule

Keeping a VM warm between runs is **not free** (the 50 s idle gap above cost 0.027 units for zero work).
Reuse saves units **only if you reuse before idle billing eats the cold-start savings**:

> **Break-even: reuse within ~one cold-start's worth of idle time** (~5 min on T4, ~20 min on L4). Reuse
> promptly → big win; let the VM sit warm for an hour → idle billing exceeds the saved cold start and reuse
> goes **net negative**. On the last run, omit `--keep-session` so the VM stops immediately.

### Savings (avoiding the 2nd+ cold start)

| Scenario (2 runs, T4) | units |
|---|---:|
| 2 separate cold runs | 0.343 |
| 1 cold + 1 warm reuse | 0.225 |
| **saved** | **0.118 (34 %)** |

Savings grow with each reuse — asymptotically you stop paying the cold start at all, approaching
**cold / (cold + task)** savings: **~85 % on T4, ~69 % on L4**.

### L4 / DiffusionGemma projection

Same model, bigger numbers (L4 ≈ 4.8 units/hr; cold ~22 min, task ~10 min):

- **Cold start ≈ 1.76 units** vs **warm task ≈ 0.80 units** → each reuse avoids ~1.76 units (~¥20).
- 2 runs: ~5.1 (both cold) → ~3.4 (cold+warm) ≈ **33 % saved**; 5 runs ≈ **55 % saved**.

(Projected — a live L4 measurement is blocked today by the vLLM-nightly regression below. The T4 rate
1.96/hr is the commonly-cited figure; the **ratios** are rate-independent, and the authoritative per-account
numbers are in *Colab → Usage*.)

## Caveats

- **Idle billing** — see the break-even rule above. Reuse promptly; tear down on the last run.
- **VM survival window** — between separate launcher invocations the VM survives Colab's idle policy
  (~tens of minutes), not hours. Reuse for same-sitting iteration, not next-day.
- **Serve-affecting config changes need a fresh cold start** — reuse does not restart the running serve.
- **DiffusionGemma/L4 cold start is currently blocked by a vLLM-nightly regression** (2026-06-24: `vllm
  --pre` resolved to `v0.23.0`, whose engine core rejects `DiffusionGemmaDecoderModel.forward` —
  `input_ids not found`; 2026-06-23 used `v0.23.1rc1.dev307` and worked). This is upstream, not the reuse
  mechanism. Pin vLLM to a compatible build for reproducibility. The fee-free T4 (llama.cpp/Ollama,
  prebuilt-wheel) paths are immune.
