---
name: colab-openclaw-diffusiongemma
description: Run a Colab CLI-controlled OpenClaw appliance where vLLM serves quantized DiffusionGemma inside the same Colab GPU runtime, then collect prompt outputs and logs.
allowed-tools: Bash Read Write Edit Glob Grep
---

# Colab OpenClaw DiffusionGemma skill

Use this skill for a job-oriented local Claude Code workflow that provisions a Colab GPU, starts vLLM, configures OpenClaw to use the Colab-local vLLM endpoint, runs a prompt through OpenClaw's inference CLI, and downloads a single result bundle.

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

The standard `bin/` workflow above is the intended product, but it has **not** yet been made
to pass end-to-end on this account; the validated path is the dev harness. Full details in
`docs/validation_findings.md`. Key points:

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
- **GPU reality:** L4 is **not entitled** on this free account, so DiffusionGemma-26B cannot
  run here — only the T4 small-model (`Qwen/Qwen2.5-0.5B-Instruct`) validation. DiffusionGemma
  needs Colab Pro/Enterprise or a rented L4/A100.
