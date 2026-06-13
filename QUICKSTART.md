# Quickstart

## 1. Install local prerequisites

```bash
python -m pip install --upgrade google-colab-cli
colab version
```

Authenticate the CLI using the flow appropriate for your machine:

```bash
colab sessions
```

## 2. Validate the bundle locally

```bash
python scripts/self_test.py
bash -n bin/colab_openclaw_diffusiongemma.sh
```

## 3. Run a smoke test

```bash
bash bin/colab_openclaw_diffusiongemma.sh \
  --session openclaw-dg-smoke \
  --gpu T4 \
  --config configs/smoke_test_tiny.json \
  --task examples/prompt_task.json \
  --out ./runs/smoke
```

## 4. Run DiffusionGemma NVFP4

```bash
bash bin/colab_openclaw_diffusiongemma.sh \
  --session openclaw-dg \
  --gpu L4 \
  --config configs/diffusiongemma_nvfp4.json \
  --task examples/prompt_task.json \
  --out ./runs/diffusiongemma
```

Use A100/H100 class hardware when available for the most reliable full-checkpoint deployment. If the Colab GPU is too small, the remote manifest and logs will capture the failure mode.

## 5. Keep or stop the session

By default, the script stops the session after artifact collection. Add `--keep-session` when you want to inspect the runtime:

```bash
bash bin/colab_openclaw_diffusiongemma.sh --keep-session ...
```
