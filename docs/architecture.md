# Architecture

The default architecture is a job-oriented appliance controlled by Claude Code through Google Colab CLI.

```text
Local Claude Code
  -> Bash
  -> Google Colab CLI
  -> one Colab GPU runtime
       -> vLLM OpenAI-compatible server on 127.0.0.1:8000/v1
       -> OpenClaw configured with provider id vllm
       -> OpenClaw Gateway on loopback
       -> OpenClaw infer prompt probe
       -> results zip
```

The Colab runtime is not treated as a durable server. It is a temporary job executor. The local script downloads artifacts and session logs after each run.

## Why Colab CLI instead of a public Gateway URL?

The default control path avoids exposing the OpenClaw Gateway to the public internet. Claude Code sends scripts and JSON through Colab CLI. A remote Python script inside the Colab VM talks to OpenClaw and vLLM over localhost.

## Main files

- `bin/colab_openclaw_diffusiongemma.sh`: local controller
- `remote/remote_colab_openclaw_diffusiongemma.py`: remote Colab appliance script
- `configs/diffusiongemma_nvfp4.json`: full quantized DiffusionGemma config
- `configs/smoke_test_tiny.json`: small model smoke-test config
- `examples/prompt_task.json`: prompt task
