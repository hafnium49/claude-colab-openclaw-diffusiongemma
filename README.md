# Claude Code Colab OpenClaw + DiffusionGemma Appliance

This repository is a Claude Code-native project scaffold for controlling a single Google Colab GPU runtime through the Google Colab CLI. The Colab runtime hosts:

- a vLLM OpenAI-compatible server for `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`
- an OpenClaw Gateway configured to use that local vLLM endpoint
- a headless prompt runner that collects outputs and logs into one zip archive

The local machine runs Claude Code and uses `colab` as the transport. There is no public OpenClaw tunnel requirement in the default workflow.

## Scope

This bundle is only for a Colab-hosted OpenClaw + DiffusionGemma language-model appliance. It intentionally does not include unrelated workloads.

## Quick start

Install Google Colab CLI locally, authenticate it, then run:

```bash
unzip claude-colab-openclaw-diffusiongemma.zip
cd claude-colab-openclaw-diffusiongemma
bash bin/colab_openclaw_diffusiongemma.sh \
  --session openclaw-dg \
  --gpu L4 \
  --config configs/diffusiongemma_nvfp4.json \
  --task examples/prompt_task.json \
  --out ./runs/openclaw-dg
```

For a lightweight architecture smoke test, use:

```bash
bash bin/colab_openclaw_diffusiongemma.sh \
  --session openclaw-dg-smoke \
  --gpu T4 \
  --config configs/smoke_test_tiny.json \
  --task examples/prompt_task.json \
  --out ./runs/smoke
```

The output directory receives:

- `openclaw_diffusiongemma_results.zip`
- `colab_session_log.ipynb`
- downloaded `manifest.json` when available
- local command logs

## Claude Code integration

Copy or keep the included project-native files:

```text
.claude/agents/colab-openclaw-diffusiongemma.md
.claude/skills/colab-openclaw-diffusiongemma/SKILL.md
```

Then ask Claude Code to use the `colab-openclaw-diffusiongemma` subagent, or invoke the skill directly with:

```text
/colab-openclaw-diffusiongemma run configs/diffusiongemma_nvfp4.json examples/prompt_task.json
```

## Hardware note

The default quantized DiffusionGemma config is intended for a high-memory Colab GPU. Use the smoke-test config first to validate Colab CLI, OpenClaw, and vLLM orchestration before spending time on the full checkpoint.
