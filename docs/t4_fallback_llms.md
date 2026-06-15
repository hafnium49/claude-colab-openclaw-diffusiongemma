# T4 fallback LLMs — agent-model picks

> Context for this repo: `DiffusionGemma-26B-A4B-NVFP4` needs a ≥~24 GB NVIDIA GPU (NVFP4 is
> a Blackwell-only format and won't run on TPU), which free Colab doesn't provide. When the
> target model isn't reachable, these are the small **agent** LLMs to serve via vLLM on a
> Colab **T4** instead of the `Qwen2.5-0.5B` placeholder used in the validation harness. Swap
> `MODEL` in `runs/dev/*_boot.py` and keep the OpenClaw infer fixes (`requiresStringContent`,
> `maxTokens` < `--max-model-len`). See also `docs/validation_findings.md`.

There is **no perfect public apples-to-apples table** for exactly these three target models:

```text
LFM2.5-8B-A1B
Gemma 4 12B-it
Qwen3.5-9B
```

The closest agent benchmarks are **BFCL-V4** for tool/function calling and **τ² / TAU2-Bench**
for multi-turn tool-using conversational agents. BFCL-V4 explicitly evaluates function/tool-call
accuracy, while τ²-Bench evaluates agents in airline, retail, and telecom-style multi-turn
environments.

## 1. Best target-size comparison: 8–14B-class models

| Model              |                  Params |                       BFCL-V4 |                                                    TAU2 / τ² | Agent interpretation                                                              |
| ------------------ | ----------------------: | ----------------------------: | -----------------------------------------------------------: | --------------------------------------------------------------------------------- |
| **Qwen3.5-9B**     |                9B dense |                      **66.1** |                                                     **79.1** | Strongest overall public agent score in this size band                            |
| **Gemma 4 12B-it** |               12B dense | not reported in official card |                                      **69.0** average over 3 | Good multi-turn agent score; BFCL-V4 gap makes tool-calling comparison incomplete |
| **LFM2.5-8B-A1B**  | 8B total / ~1.5B active |                     **49.73** | Telecom **88.07**, Retail **39.82**; 2-domain mean **63.95** | Very strong telecom, weak retail; optimized for low-latency local agents          |

- Qwen3.5-9B reports **BFCL-V4 66.1**, **TAU2-Bench 79.1**, **VITA-Bench 29.8**, and **DeepPlanning 18.0** in its model card.
- Gemma 4 12B reports **TAU2 average 69.0**, but Google's model card does **not** report BFCL-V4 for the 12B model.
- LFM2.5-8B-A1B reports **BFCL-V4 49.73**, **Tau² Telecom 88.07**, and **Tau² Retail 39.82**; Liquid does not report an airline score in the same table, so the 63.95 figure is only a two-domain proxy, not a true TAU2 average.

## 2. Strict same-table comparison from Liquid's published benchmark

Liquid's table is the cleanest same-harness comparison, but it compares **LFM2.5-8B-A1B**,
**Gemma 4 E4B**, and **Qwen3.5-4B**, not Qwen3.5-9B or Gemma 4 12B.

| Model                  |               Params |   BFCL-V3 |   BFCL-V4 | Tau² Telecom | Tau² Retail |
| ---------------------- | -------------------: | --------: | --------: | -----------: | ----------: |
| **Qwen3.5-4B**         |                   4B | **71.06** | **54.01** |        87.72 |   **71.93** |
| **LFM2.5-8B-A1B**      |             8B / A1B |     64.79 |     49.73 |    **88.07** |       39.82 |
| **Gemma 4 E4B-it**     | 8B / ~4.5B effective |     57.31 |     33.92 |        26.75 |       42.11 |
| **Gemma 4 26B-A4B-it** |            26B / A4B |     68.87 | **55.87** |        42.11 |       55.26 |

This table says something important: **small Qwen3.5 beats LFM2.5 on BFCL-V4 and retail τ²,
while LFM2.5 narrowly leads on telecom τ².** Gemma 4 E4B is weaker on these agent-specific
numbers, even though larger Gemma 4 models are strong at general reasoning and coding.

## Practical ranking for a T4 fallback agent

|  Rank | Model                     | Why                                                                                  |
| ----: | ------------------------- | ------------------------------------------------------------------------------------ |
| **1** | **Qwen3.5-9B 4-bit**      | Best public agent benchmark profile: BFCL-V4 + TAU2 + planning scores                |
| **2** | **LFM2.5-8B-A1B Q4/Q5**   | Fastest and most memory-comfortable; excellent if the router mainly dispatches tools |
| **3** | **Gemma 4 12B-it QAT Q4** | Use when multimodal ability matters; agent benchmark evidence is less complete       |

## Bottom line

For **agent AI performance**, choose:

```text
Qwen3.5-9B 4-bit
```

For **fast local tool-routing with maximum T4 stability**, choose:

```text
LFM2.5-8B-A1B GGUF Q5_K_M
```

For **vision / screenshots / multimodal local agent use**, choose:

```text
Gemma 4 12B-it QAT Q4
```

The quantitative reason is simple: **Qwen3.5-9B is the only one of the three that publicly
reports strong scores across BFCL-V4, TAU2-Bench, VITA-Bench, and DeepPlanning in the 8–14B
range.**

> T4 fit (15 GB VRAM): a 9B at 4-bit ≈ 5–6 GB, a 12B at 4-bit ≈ 7 GB — both leave room for the
> KV cache. Model names are forward-dated (Qwen3.5 / Gemma 4 / LFM2.5); verify the exact Hugging
> Face repo IDs before serving.

## References

- BFCL-V4 — Berkeley Function Calling Leaderboard: <https://gorilla.cs.berkeley.edu/leaderboard.html>
- Qwen3.5-9B model card: <https://huggingface.co/Qwen/Qwen3.5-9B>
- Gemma 4 model card: <https://ai.google.dev/gemma/docs/core/model_card_4>
- LFM2.5-8B-A1B (Liquid AI): <https://www.liquid.ai/ja/blog/lfm2-5-8b-a1b>
