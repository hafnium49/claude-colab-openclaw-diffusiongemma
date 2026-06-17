# Validation findings (live log)

Status of bringing up **OpenClaw → vLLM → (target) DiffusionGemma** on a Google Colab GPU,
controlled from local Claude Code via the `colab` CLI. This file records what has actually
been **run and proven** on Colab, as opposed to the aspirational design in `architecture.md`.

Last updated: 2026-06-15. Account: free-tier consumer Colab (`hafnium49@gmail.com`).

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
cold start can complete.

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

## Open items

- [x] Land `infer_ok=true` on T4 via the decoupled harness — **done, run #6, 2026-06-15.**
- [ ] Refactor `bin/` + `remote/` from detached-bootstrap+sparse-poll to the short-exec model
      (port `e2e_boot.py`/`e2e_poll.py`/`e2e_finish.py` into the launcher; update `self_test.py`).
- [ ] Obtain an L4/A100 and run the real DiffusionGemma profile (only remaining blocker is GPU).
