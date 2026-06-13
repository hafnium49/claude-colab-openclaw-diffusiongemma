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
