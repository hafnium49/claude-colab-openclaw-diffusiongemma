#!/usr/bin/env python3
"""Generate notebooks/openclaw_chat_colab.ipynb from the proven vLLM+OpenClaw recipe.
Authoring .ipynb JSON by hand is error-prone; build it with json.dump so it's always valid.

This is the self-contained, browser-run path: open it in Colab, Run all, chat. The open tab
supplies the runtime heartbeat (so the VM stays alive — the headless CLI couldn't), and there
is NO Cloudflare, NO tunnel, NO copy-pasting snippets. Each chat cell is self-contained so it
can't hit the cross-cell NameError."""
import json, os

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Chat with OpenClaw + vLLM on Colab (single instance, no tunnel)

Run this notebook **in Colab** to host an LLM (vLLM) + **OpenClaw** in one GPU instance and
chat with it from notebook cells. No Cloudflare, no GitHub, no copy-pasting — just **Run all**.

**Setup (do this first):**
1. `Runtime → Change runtime type → T4 GPU`  →  Save
2. `Runtime → Run all`  (or run the cells top to bottom)
3. Keep this tab open — that open tab is what keeps the VM alive.

The first run installs vLLM + OpenClaw and does a one-time ~7-min model warmup. After that,
go to the **Chat** cell near the bottom, edit `MESSAGE`, and re-run it for each turn.""")

md("""## 📋 Briefing — what this is and why

**Goal:** host an LLM (**vLLM**) + **OpenClaw** (an agent gateway + chat UI) inside **one Colab
GPU instance** and chat with it — controlled entirely from this notebook, no external services.

**Why a small model here, not DiffusionGemma?** The original target,
`diffusiongemma-26B-A4B-it-NVFP4`, is quantized in **NVFP4** — an NVIDIA *Blackwell-only*
format. It needs a ≥~24 GB NVIDIA GPU (L4 / A100 / Blackwell) and **cannot run on a free Colab
T4 or on a TPU** (TPUs can't execute NVFP4; a full-precision 26 B won't fit a single TPU chip
either). So this notebook runs a **T4-fittable** model. For stronger small agent models, see
`docs/t4_fallback_llms.md` (e.g. Qwen3.5-9B 4-bit).

**What's proven:** the full path — vLLM serve → OpenClaw gateway → inference — is validated
green on a Colab T4. Two non-obvious OpenClaw fixes are baked into cell 4 (without them the
gateway returns an empty `incomplete_result`): force **string content**, and keep the model's
**`maxTokens` below `--max-model-len`** so the request doesn't overflow the context window.

**About keeping the VM alive:** free Colab reclaims idle runtimes, and the `colab` CLI's
keep-alive had a bug (403 on a project-scoped API → quit in ~72 s; fixed in google-colab-cli
≥ 0.5.12). Running this as a **notebook with the tab open is the robust path** — your browser
frontend is the heartbeat, so the VM lives for hours of active use. Just don't close the tab.

**Cost/limits:** it's an ephemeral T4 — nothing persists after the runtime ends, and heavy
same-day use can shorten how long Colab lets a VM run. Treat each session as disposable.

---""")

md("### 1 — Install vLLM (CUDA-13 fix) + OpenClaw  *(~4 min)*")
code("""# vLLM 0.23 needs CUDA 13; Colab ships torch+cu128 -> remove it, reinstall via uv (+cu130).
# vLLM runs as a subprocess later, so no kernel restart is needed here.
import subprocess, sys
print("Installing uv + vLLM (cu13) ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "uv"], check=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y",
                "torch", "torchvision", "torchaudio"])
subprocess.run("uv pip install --system --torch-backend auto vllm", shell=True, check=True)
print("Installing OpenClaw (npm-based installer) ...", flush=True)
subprocess.run("curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard",
               shell=True, check=True)
print("\\nInstall complete.")""")

md("""### 2 — Pick the model + a gateway token

`Qwen/Qwen2.5-3B-Instruct` is a solid, real chat model that fits T4's 15 GB. Swap `MODEL` for
something lighter (`Qwen/Qwen2.5-0.5B-Instruct`, fastest) or a stronger agent model from
`docs/t4_fallback_llms.md` (e.g. a Qwen3.5-9B 4-bit build) — just confirm the HF repo id exists.""")
code("""import os, secrets
MODEL = "Qwen/Qwen2.5-3B-Instruct"     # <- change me if you like
os.environ["VLLM_API_KEY"] = "vllm-local"
os.environ["OPENCLAW_GATEWAY_TOKEN"] = secrets.token_hex(8)
open("/content/oc_token.txt", "w").write(os.environ["OPENCLAW_GATEWAY_TOKEN"])  # chat cells read this
print("MODEL =", MODEL)
print("gateway token saved to /content/oc_token.txt")""")

md("### 3 — Start vLLM and wait for warmup  *(~7 min on T4, one time)*")
code("""import subprocess, glob, os, time, urllib.request
# LD_LIBRARY_PATH -> the nvidia pip libs (provides libcudart.so.13 for the cu13 build).
nvlibs = ":".join(sorted(glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib")))
env = dict(os.environ, LD_LIBRARY_PATH=nvlibs + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
subprocess.Popen(["bash", "-c",
    f"vllm serve {MODEL} --host 127.0.0.1 --port 8000 --max-model-len 8192 "
    f"--enforce-eager > /content/vllm.log 2>&1"], env=env)

print("vLLM starting (downloads weights + warms up; ~7 min on T4) ...", flush=True)
req = urllib.request.Request("http://127.0.0.1:8000/v1/models",
                             headers={"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]})
t0 = time.time(); ready = False
while time.time() - t0 < 1200:
    try:
        urllib.request.urlopen(req, timeout=4); ready = True; break
    except Exception:
        print(f"   warming {int(time.time()-t0)}s ...", flush=True)
        subprocess.run("tail -1 /content/vllm.log", shell=True)
        time.sleep(12)
print(("vLLM READY after %ds" % (time.time()-t0)) if ready else
      "TIMEOUT — check /content/vllm.log (out-of-memory? try a smaller MODEL)")""")

md("### 4 — Configure OpenClaw against vLLM and start the gateway")
code("""import subprocess, os
PATHX = 'export PATH="$(npm prefix -g)/bin:$PATH"; '   # OpenClaw lives in the npm global bin
def oc(c): return subprocess.run(["bash", "-c", PATHX + c], capture_output=True, text=True)

ob = oc("openclaw onboard --non-interactive --accept-risk --mode local "
        "--auth-choice custom-api-key --custom-provider-id vllm "
        "--custom-base-url http://127.0.0.1:8000/v1 "
        f"--custom-model-id {MODEL} --custom-compatibility openai "
        '--custom-api-key \"$VLLM_API_KEY\" --custom-text-input '
        "--gateway-port 18789 --gateway-bind loopback --gateway-auth token "
        "--gateway-token-ref-env OPENCLAW_GATEWAY_TOKEN "
        "--skip-daemon --skip-skills --skip-channels --skip-health --skip-ui --json")
print("onboard rc =", ob.returncode)

# Two fixes so the gateway infer actually returns text on a local OpenAI-compat backend:
#  - requiresStringContent: send plain-string content (not an array) -> avoids empty completion
#  - maxTokens < --max-model-len -> avoids the context-overflow "incomplete_result"
for k, v in [("compat.requiresStringContent", "true"), ("compat.supportsTools", "false"),
             ("maxTokens", "1024"), ("contextWindow", "8192")]:
    oc(f"openclaw config set 'models.providers.vllm.models[0].{k}' {v}")

subprocess.Popen(["bash", "-c", PATHX + "openclaw gateway run > /content/gateway.log 2>&1"])
import time; time.sleep(12)
print("gateway status:")
print(oc("openclaw gateway status").stdout[-300:] or "(started)")""")

md("""### 5 — 💬 Chat

Edit `MESSAGE` and run this cell. **Re-run it (with a new `MESSAGE`) for each turn.** This cell
is self-contained — it reads the token from disk and calls the model in one shot, so it never
depends on another cell having defined a function.""")
code("""MESSAGE = "Hello! Tell me a fun fact in one short sentence."

import subprocess, json
_tok = open("/content/oc_token.txt").read().strip()
_cmd = ('export PATH="$(npm prefix -g)/bin:$PATH"; '
        f'export OPENCLAW_GATEWAY_TOKEN={_tok}; '
        f'openclaw infer model run --gateway --model vllm/{MODEL} --prompt {MESSAGE!r} --json')
_r = subprocess.run(["bash", "-c", _cmd], capture_output=True, text=True)
try:
    print(json.loads(_r.stdout)["outputs"][0]["text"])
except Exception:
    print(_r.stdout or _r.stderr)""")

md("""### 6 — (Optional) Open the OpenClaw browser Control UI inside Colab

Surfaces OpenClaw's web UI through Colab's own port proxy (no tunnel). When it asks for a
token, paste the value printed below.""")
code("""from google.colab import output
import os
print("Paste this token into the OpenClaw UI:", open("/content/oc_token.txt").read().strip())
output.serve_kernel_port_as_window(18789, path="/")
# Inline instead of a new window? use:
# output.serve_kernel_port_as_iframe(18789, path="/", height="720")""")

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
