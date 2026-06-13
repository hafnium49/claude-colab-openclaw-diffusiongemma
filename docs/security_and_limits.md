# Security and limits

## Secrets

Do not commit Hugging Face tokens, OpenClaw tokens, or any provider credentials. Pass secrets through environment variables or Colab-side secret management.

## Network exposure

The default workflow keeps both services on loopback inside Colab:

```text
vLLM: http://127.0.0.1:8000/v1
OpenClaw Gateway: loopback, port 18789
```

Avoid public tunnels unless you intentionally accept the risk.

## Runtime lifetime

Colab runtimes are temporary. This project assumes batch-style runs that download artifacts after completion.

## GPU memory

The full DiffusionGemma NVFP4 profile may require a high-memory GPU and compatible vLLM build. Run `configs/smoke_test_tiny.json` first to validate the orchestration path.
