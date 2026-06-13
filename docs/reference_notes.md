# Reference notes

- OpenClaw uses `vllm` as a local OpenAI-compatible provider at `http://127.0.0.1:8000/v1` when configured that way.
- OpenClaw `infer model run` is the headless provider-backed prompt surface used by the remote script.
- Google Colab CLI is the local-to-remote transport for provisioning, execution, file upload/download, logs, and teardown.
- `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` is the default quantized DiffusionGemma checkpoint in this bundle.

These notes are intentionally brief; check the upstream docs before productionizing the workflow.
