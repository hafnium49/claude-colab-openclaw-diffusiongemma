# Serving Qwen3.5-9B on a Colab T4 — use llama.cpp, not FlashInfer/vLLM

## The finding (validated 2026-06-15)

On a Colab **T4 (Turing, sm_75)**, vLLM with the default **FlashInfer** attention backend
**crashes** on the first real request for ≥3B Qwen models:

```
RuntimeError: Check failed (status == cudaSuccess): BatchPrefillWithPagedKVCache failed: invalid argument
→ vLLM shuts down
```

This is **not** a memory problem — a 4-bit 9B (~6 GB) fits T4's 15 GB. It's tied to
**Turing/sm_75 + attention backend + head_dim/paged-KV kernel shape**. Only tiny models
(Qwen2.5-0.5B, head_dim 64) survive the FlashInfer path on T4. FA2 doesn't support Turing
(Ampere+); FlashInfer has known sm_75 prefill issues. So:

```
T4 + FlashInfer + new Qwen attention shapes  = fragile / crashes
T4 + llama.cpp / Triton / XFormers           = slower but stable
L4 (Ampere) + vLLM/SGLang/FlashInfer         = the real performance host
```

**Rule:** use the T4 as a *slow-but-stable fallback* host, not a high-performance host.
For high performance (and the real DiffusionGemma-26B target), use **L4 or better**.

## Preferred T4 architecture (per user, 2026-06-15)

1. **Always-on tiny router** — `Qwen3.5-0.5B` / `Qwen3-0.6B` / xLAM 1–3B. Definitely runs.
   Job is NOT reasoning: classify request → choose tool → decide whether to escalate to
   OpenClaw / DiffusionGemma / a remote/L4 LLM → return **strict JSON**. Keep context short.
2. **Opportunistic 9B reasoning model** behind a non-FlashInfer backend, in order of preference:
   - **llama.cpp GGUF** (preferred — avoids the paged-KV kernel entirely; OpenAI-compatible)
   - SGLang Triton backend (`--attention-backend triton --sampling-backend pytorch --disable-cuda-graph`)
   - vLLM **V0 + XFormers** (`VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS`) — only if your vLLM still has V0
   - **Avoid** FlashInfer on T4 for this model family.

## First thing to try on T4 — llama.cpp GGUF (OpenAI-compatible)

```bash
llama-server \
  -hf lmstudio-community/Qwen3.5-9B-GGUF:Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 -c 4096 -b 256 --host 127.0.0.1 --port 8080
# if stable: -c 8192 -b 512 ; if it crashes/gets weird: add  -fa off
```

Gives an OpenAI endpoint at `http://127.0.0.1:8080/v1`. **OpenClaw connects identically to
the vLLM path** — just point the custom provider at port 8080 instead of 8000:

```
openclaw onboard ... --custom-base-url http://127.0.0.1:8080/v1 \
  --custom-model-id Qwen3.5-9B ...   # + the requiresStringContent / maxTokens<ctx fixes
```

Don't test long context first — KV cache + kernel choice matter more than 4-bit weight size on T4.

## SGLang diagnostic alternative (if llama.cpp unavailable)

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --host 127.0.0.1 --port 30000 \
  --attention-backend triton --sampling-backend pytorch --disable-cuda-graph \
  --context-length 4096 --max-running-requests 1 --mem-fraction-static 0.70
```

## Recommendation table

| Goal | Recommendation |
|---|---|
| Most stable T4 fallback | `llama.cpp` + Qwen3.5-9B GGUF Q4_K_M, 4K–8K ctx, one request at a time |
| SGLang experiment | `--attention-backend triton --sampling-backend pytorch --disable-cuda-graph` |
| vLLM experiment | V0 + XFormers, only if vLLM still supports V0 |
| Always-on router | 0.5B–3B model, strict JSON, no long context |
| Real upgrade | **L4 24 GB or better** (also the only path to DiffusionGemma-26B-NVFP4) |
| Not worth it | v5e-1 TPU (NVFP4 is Blackwell-only; bf16 26B doesn't fit 16 GB) |

Models: [`lmstudio-community/Qwen3.5-9B-GGUF`](https://hf.co/lmstudio-community/Qwen3.5-9B-GGUF),
[`QuantTrio/Qwen3.5-9B-AWQ`](https://hf.co/QuantTrio/Qwen3.5-9B-AWQ) (vLLM/L4).
