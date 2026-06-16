#!/usr/bin/env python3
"""Generate notebooks/openclaw_chat_colab.ipynb — the NOTEBOOK COUNTERPART of the master bash
harness.

  MASTER (source of truth):  bin/colab_openclaw_diffusiongemma.sh  ->  drives
                             remote/remote_colab_openclaw_diffusiongemma.py on a Colab VM from
                             OUTSIDE, headless, via the colab CLI. Config-driven serve backend;
                             detached bootstrap / prompt / task phases polled via *_status. This
                             is what deploys autonomously.

  THIS NOTEBOOK (counterpart): the SAME phases (bootstrap -> serve + onboard; prompt; multi-step
                             task), but as in-Colab cells you Run-all. Useful for interactive
                             testing and for surfacing the OpenClaw dashboard inline (only
                             possible when *your* browser owns the runtime).

KEEP THEM IN SYNC. If you change the model, ports, wheel, OpenClaw flags, or the task logic,
change the bash master FIRST (it is the source of truth) and mirror here. Authoring .ipynb JSON
by hand is error-prone, so this builds it with json.dump (always valid).

Roadmap context: the end goal is OpenClaw in a Colab sandbox with NO LLM API fee (the LLM is
self-hosted via llama.cpp + a local GGUF — never a paid API) running long, autonomous workloads
(e.g. deep research) with no human in the loop. Self-hosting is what makes that fee-free; the
chat cell here is just a smoke test of the same stack.
"""
import json, os

# Single source for the knobs the bash harness also uses — change in lockstep with the harness.
WHEEL = "llama-cpp-python[server]==0.3.29"
WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
MODEL_REPO = "lmstudio-community/Qwen3.5-9B-GGUF"
MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
MODEL_ID = "Qwen3.5-9B"
LLM_PORT = "8000"
GW_PORT = "18789"

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(f"""# OpenClaw + self-hosted {MODEL_ID} on Colab — notebook counterpart of the master bash harness

**The bash master is the source of truth.** This notebook mirrors, as Run-all cells, what
`bin/colab_openclaw_diffusiongemma.sh` does headlessly from outside (driving
`remote/remote_colab_openclaw_diffusiongemma.py` via the `colab` CLI): the detached
`bootstrap` → `prompt`/`task` phases. Use the bash master for the real autonomous deployment;
use this notebook for interactive testing and for the inline dashboard.

**Setup:** `Runtime → Change runtime type → T4 GPU` → Save, then `Runtime → Run all`. Keep the
tab open (your browser is the runtime heartbeat). First run installs everything + downloads the
{MODEL_FILE} model (~6 min, one time).""")

md(f"""## 📋 Briefing — what & why

**Goal / roadmap.** Run OpenClaw in a Colab sandbox with **no LLM API fee** and have it perform
long, autonomous jobs (e.g. **deep research**) **without human tasks**. The fee-free part comes
from **self-hosting the LLM** (llama.cpp serving a local GGUF on loopback) instead of calling a
paid API. This notebook stands up that exact stack; the chat cell is a smoke test of it.

**Why llama.cpp, not vLLM.** On a Colab **T4** (Turing/sm_75) vLLM's FlashInfer backend
**crashes** on ≥3B Qwen models, so the agent-grade floor model **{MODEL_ID}** can't be served via
vLLM there. llama.cpp has no paged-KV kernel and serves it (~35 tok/s, 4-bit). The prebuilt
`llama-cpp-python` CUDA wheel avoids any on-VM compile. See `docs/t4_llama_cpp_serving.md`.

**Containment.** Everything is loopback: llama.cpp on `127.0.0.1:{LLM_PORT}`, OpenClaw gateway on
`127.0.0.1:{GW_PORT}`. Nothing is exposed off the VM; the Colab runtime is the sandbox.

**Phase map (this notebook ⟷ the `bin/` master).** `bin/colab_openclaw_diffusiongemma.sh` drives
`remote/remote_colab_openclaw_diffusiongemma.py` on a Colab VM via the `colab` CLI; this notebook
runs the same phases *in-cell* on your own runtime.

| Notebook cell | Master phase (remote action) |
|---|---|
| 1 install + 2 serve | `bootstrap` → `install_llama_cpp` + `start_llama_cpp` (serve :8000) |
| 3 onboard + gateway | `bootstrap` → `configure_openclaw` (compat fixes) + `start_openclaw_gateway` |
| 4 chat (smoke) | `prompt` → `_prompt_run` infer |
| 5 autonomous task | `task` → `_task_run` (multi-step `steps[]` → `research_result.md`) |
| 6 dashboard | notebook-only (works because your browser owns the runtime) |

---""")

md("### 1 — Install llama.cpp server (prebuilt CUDA wheel) + OpenClaw  *(master: `bootstrap` → install)*")
code(f"""import subprocess, sys
print("Installing {WHEEL} (prebuilt CUDA wheel, no compile) ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "{WHEEL}",
                "--extra-index-url", "{WHEEL_INDEX}", "--prefer-binary"], check=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "huggingface_hub"], check=True)
print("Installing OpenClaw (npm-based installer) ...", flush=True)
subprocess.run("curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard",
               shell=True, check=True)
print("\\nInstall complete.")""")

md(f"### 2 — Download {MODEL_ID} GGUF + serve llama.cpp on :{LLM_PORT} + wait  *(master: `bootstrap` → `start_llama_cpp`)*")
code(f"""import subprocess, sys, os, time, urllib.request
from huggingface_hub import hf_hub_download
print("Downloading GGUF (5.6 GB, one time) ...", flush=True)
gguf = hf_hub_download("{MODEL_REPO}", "{MODEL_FILE}", local_dir="/content/gguf")
print("GGUF at", gguf, flush=True)

# OpenAI-compatible server on loopback :{LLM_PORT} (NOT :8080 — Colab's node service owns 8080).
# n_gpu_layers 99 -> full offload to the T4.
subprocess.Popen([sys.executable, "-m", "llama_cpp.server",
    "--model", gguf, "--model_alias", "{MODEL_ID}",
    "--n_gpu_layers", "99", "--n_ctx", "4096",
    "--host", "127.0.0.1", "--port", "{LLM_PORT}"],
    stdout=open("/content/llama.log", "w"), stderr=subprocess.STDOUT)

print("Loading model onto the T4 (~1 min) ...", flush=True)
t0 = time.time(); ready = False
while time.time() - t0 < 600:
    try:
        urllib.request.urlopen("http://127.0.0.1:{LLM_PORT}/v1/models", timeout=4); ready = True; break
    except Exception:
        print(f"   loading {{int(time.time()-t0)}}s ...", flush=True)
        subprocess.run("tail -1 /content/llama.log", shell=True); time.sleep(8)
print(("llama.cpp READY after %ds" % (time.time()-t0)) if ready else
      "TIMEOUT — check /content/llama.log (OOM? try a smaller model)")""")

md(f"### 3 — Onboard OpenClaw against :{LLM_PORT} + start gateway  *(master: `bootstrap` → `configure_openclaw` + gateway)*")
code(f"""import subprocess, shutil, os, time
# Resolve openclaw by ABSOLUTE path -> never 'openclaw: command not found' (npm symlinks it into
# the global bin, e.g. /usr/bin). Provider id kept 'vllm' so it matches the bash harness + chat.
OPENCLAW = shutil.which("openclaw") or "/usr/bin/openclaw"
os.environ.setdefault("OPENCLAW_GATEWAY_TOKEN", "llama-local-token")
def oc(args): return subprocess.run([OPENCLAW] + args, capture_output=True, text=True)
print("openclaw:", OPENCLAW)

ob = oc(["onboard", "--non-interactive", "--accept-risk", "--mode", "local",
    "--auth-choice", "custom-api-key", "--custom-provider-id", "vllm",
    "--custom-base-url", "http://127.0.0.1:{LLM_PORT}/v1",
    "--custom-model-id", "{MODEL_ID}", "--custom-compatibility", "openai",
    "--custom-api-key", "llama-local", "--custom-text-input",
    "--gateway-port", "{GW_PORT}", "--gateway-bind", "loopback", "--gateway-auth", "token",
    "--gateway-token-ref-env", "OPENCLAW_GATEWAY_TOKEN",
    "--skip-daemon", "--skip-skills", "--skip-channels", "--skip-health", "--skip-ui", "--json"])
print("onboard rc =", ob.returncode)

# Infer fixes (mirror the bash harness): string content; maxTokens < n_ctx -> avoid empty/overflow.
for k, v in [("compat.requiresStringContent", "true"), ("compat.supportsTools", "false"),
             ("maxTokens", "1024"), ("contextWindow", "4096")]:
    oc(["config", "set", f"models.providers.vllm.models[0].{{k}}", v])

# Gateway runs in-process (lives while the tab is open) — needed only for the inline dashboard.
subprocess.Popen([OPENCLAW, "gateway", "run"],
    stdout=open("/content/gateway.log", "w"), stderr=subprocess.STDOUT)
time.sleep(12)
print("gateway started on 127.0.0.1:{GW_PORT}")""")

md(f"""### 4 — 💬 Chat (smoke test)  *(master: `prompt` → `_prompt_run`)*

Self-contained: resolves `openclaw` by absolute path and uses **direct** infer (no gateway
needed). Edit `MESSAGE`, re-run per turn. {MODEL_ID} is a reasoning model — a plain message takes
~1–2 min and shows a `<think>` trace; prefix `/no_think` for a fast, clean answer.""")
code(f'''MESSAGE  = "Hello! Who are you, in one sentence?"   # <- edit me; re-run for each turn
MODEL_ID = "{MODEL_ID}"

import subprocess, json, shutil
OPENCLAW = shutil.which("openclaw") or "/usr/bin/openclaw"
r = subprocess.run([OPENCLAW, "infer", "model", "run", "--model", f"vllm/{{MODEL_ID}}",
                    "--prompt", MESSAGE, "--json"], capture_output=True, text=True)
try:
    print(json.loads(r.stdout)["outputs"][0]["text"])
except Exception:
    print(r.stdout or r.stderr)''')

md("""### 5 — 🤖 Autonomous task  *(master: `task` → `_task_run`)* — multi-step deep research, no human

Mirrors the master's `task` worker: runs a **list of `STEPS`** sequentially through OpenClaw (one
self-hosted-LLM call each — no paid API) and accumulates the answers into
`/content/research_result.md`, exactly like
`bin/colab_openclaw_diffusiongemma.sh --task examples/research_task.json`. Edit `TOPIC` / `STEPS`.

The master runs this **detached** (to survive the colab-exec window); here it runs **in-cell**
because your open tab owns the runtime. The `infer()` salvage rule matches the master's
`extract_infer_text` (only a non-empty `outputs[0].text` counts). For true tool-using research,
re-onboard in cell 3 **without** `--skip-skills`.""")
code('''import subprocess, json, shutil
OPENCLAW  = shutil.which("openclaw") or "/usr/bin/openclaw"
MODEL_ID  = "Qwen3.5-9B"
TRANSPORT = "local"   # direct infer (no --gateway) — the robust path the master uses for research

TOPIC = "Tradeoffs of self-hosting an LLM on a single GPU vs. paid inference APIs."
STEPS = [
    "List the main dimensions to compare self-hosted single-GPU LLM inference against paid APIs (cost, latency, throughput, privacy, control, ops burden, capability ceiling). One line each.",
    "For a single 16GB GPU (e.g. a T4) running a 4-bit ~9B model, give concrete tokens/sec, context limits, and what breaks at higher load.",
    "When does a paid API clearly win, and when does self-hosting clearly win? Give 3 scenarios each.",
    "Synthesize the above into a 5-bullet executive summary with a final recommendation heuristic.",
]

def infer(prompt):
    flag = ["--gateway"] if TRANSPORT == "gateway" else []   # else: direct infer, the robust path
    r = subprocess.run([OPENCLAW, "infer", "model", "run", *flag, "--model", f"vllm/{MODEL_ID}",
                        "--prompt", prompt, "--json"], capture_output=True, text=True)
    try:    # same salvage rule as the master's extract_infer_text: only non-empty text counts
        outs = (json.loads(r.stdout) or {}).get("outputs") or []
        if outs and isinstance(outs[0], dict) and isinstance(outs[0].get("text"), str) and outs[0]["text"].strip():
            return outs[0]["text"]
    except Exception:
        pass
    return None

lines = [f"# Autonomous research result\\n\\n- Topic: {TOPIC}\\n- Model: vllm/{MODEL_ID}\\n"]
for i, step in enumerate(STEPS, 1):
    print(f"step {i}/{len(STEPS)}: {step[:60]}...", flush=True)
    text = infer(step)
    lines.append(f"\\n## Step {i}\\n\\n**Prompt:** {step}\\n\\n{text or '(no text returned)'}\\n")
open("/content/research_result.md", "w").write("\\n".join(lines))
print("\\nWrote /content/research_result.md (" + str(len(STEPS)) + " steps)\\n" + "=" * 60)
print(open("/content/research_result.md").read())''')

md(f"""### 6 — (Optional) OpenClaw Control dashboard, inline — no tunnel

This works **only because your browser owns this runtime** (it's the Colab frontend), so
`serve_kernel_port_as_iframe` can mint a Google-authenticated proxy to the gateway's port
{GW_PORT} — no public tunnel. (This is exactly why the dashboard is NOT reachable when a headless
CLI manages the VM.) Requires the gateway from cell 3 to be running.""")
code(f'''from google.colab import output
import os
print("If the dashboard asks for a token, paste:", os.environ.get("OPENCLAW_GATEWAY_TOKEN", "llama-local-token"))
output.serve_kernel_port_as_iframe({GW_PORT}, path="/", height="720")''')

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openclaw_chat_colab.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
