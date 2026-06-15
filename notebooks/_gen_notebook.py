#!/usr/bin/env python3
"""Generate notebooks/openclaw_chat_colab.ipynb from the proven vLLM+OpenClaw recipe.
Authoring .ipynb JSON by hand is error-prone; build it with json.dump so it's always valid.
This is the UI-in-Colab, no-cloudflare path: a browser-run notebook (the open tab keeps the
runtime alive, which the headless CLI couldn't) that surfaces the OpenClaw Control UI via
Colab's native port proxy."""
import json, os

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Chat with OpenClaw + vLLM on Colab — no tunnel

This runs **one Colab GPU instance** hosting **vLLM** (OpenAI-compatible LLM server) and
**OpenClaw** (gateway + browser Control UI), then surfaces the OpenClaw UI **inside Colab**
via Colab's built-in port proxy — **no Cloudflare / no public tunnel**.

**Before you start:** `Runtime > Change runtime type > T4 GPU`, then run the cells top to bottom.

**Notes**
- Keep this browser tab open — that's what keeps the runtime alive (the headless CLI couldn't).
- First run does a ~7-min vLLM cold start on T4 (one time). After that, chat as long as the tab is open.
- Default model is tiny (`Qwen2.5-0.5B`) so it's fast/reliable; for much better chat, change
  `MODEL` in cell 2 to `Qwen/Qwen2.5-3B-Instruct` (fits T4's 15 GB) — slightly longer warmup.
- DiffusionGemma-26B still needs an L4 (24 GB), which free Colab doesn't grant; this is the
  small-model chat path.""")

md("### 1 — Install vLLM (CUDA-13 fix) + OpenClaw  *(~3-4 min)*")
code("""# vLLM 0.23 needs CUDA 13; Colab ships torch+cu128 -> remove it, reinstall via uv (+cu130).
import subprocess, sys
print("Installing uv + vLLM (cu13) ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "uv"], check=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y",
                "torch", "torchvision", "torchaudio"])
subprocess.run("uv pip install --system --torch-backend auto vllm", shell=True, check=True)
print("Installing OpenClaw (npm-based installer) ...", flush=True)
subprocess.run("curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard",
               shell=True, check=True)
print("\\nInstall done.")""")

md("### 2 — Pick the model + a gateway token")
code("""import os, secrets
# T4 has ~15 GB. 0.5B = fast & proven. For better chat: "Qwen/Qwen2.5-3B-Instruct".
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
os.environ["VLLM_API_KEY"] = "vllm-local"
os.environ["OPENCLAW_GATEWAY_TOKEN"] = secrets.token_hex(8)
print("MODEL =", MODEL)
print("Gateway token (you'll paste this into the OpenClaw UI):", os.environ["OPENCLAW_GATEWAY_TOKEN"])""")

md("### 3 — Start vLLM and wait for it to warm up  *(~7 min on T4, one time)*")
code("""import subprocess, glob, os, time, urllib.request
# LD_LIBRARY_PATH -> the nvidia pip libs (provides libcudart.so.13 for the cu13 build).
nvlibs = ":".join(sorted(glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib")))
env = dict(os.environ, LD_LIBRARY_PATH=nvlibs + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
subprocess.Popen(["bash", "-c",
    f"vllm serve {MODEL} --host 127.0.0.1 --port 8000 --max-model-len 8192 "
    f"--enforce-eager > /content/vllm.log 2>&1"], env=env)

print("vLLM starting (T4 cold start ~7 min: model load + FlashInfer warmup) ...", flush=True)
req = urllib.request.Request("http://127.0.0.1:8000/v1/models",
                             headers={"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]})
t0 = time.time(); ready = False
while time.time() - t0 < 900:
    try:
        urllib.request.urlopen(req, timeout=4); ready = True; break
    except Exception:
        print(f"   warming {int(time.time()-t0)}s ... (tail of vllm.log below)", flush=True)
        subprocess.run("tail -1 /content/vllm.log", shell=True)
        time.sleep(12)
print(("vLLM READY after %ds" % (time.time()-t0)) if ready else "TIMEOUT - check /content/vllm.log")""")

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

# Two fixes that make the gateway infer actually return text on a local OpenAI-compat backend:
#  - requiresStringContent: send plain-string content (not an array) -> avoids empty completion
#  - maxTokens < --max-model-len -> avoids the context-overflow "incomplete_result"
for k, v in [("compat.requiresStringContent", "true"), ("compat.supportsTools", "false"),
             ("maxTokens", "1024"), ("contextWindow", "8192")]:
    oc(f"openclaw config set 'models.providers.vllm.models[0].{k}' {v}")

subprocess.Popen(["bash", "-c", PATHX + "openclaw gateway run > /content/gateway.log 2>&1"])
import time; time.sleep(12)
print("gateway status:")
print(oc("openclaw gateway status").stdout[-400:])""")

md("""### 5 — Open the OpenClaw Control UI inside Colab  *(no Cloudflare)*

This uses Colab's built-in port proxy to expose the gateway's Control UI through Colab's own
HTTPS infrastructure (authenticated by your Google login). Click the link it prints; when the
UI asks for a token/password, paste the **gateway token** from cell 2.""")
code("""from google.colab import output
import os
print("Gateway token to paste into the OpenClaw UI:  ", os.environ["OPENCLAW_GATEWAY_TOKEN"])
print("Opening the OpenClaw Control UI (new window) ...")
output.serve_kernel_port_as_window(18789, path="/")
# Prefer it embedded inline instead? comment the line above and use:
# output.serve_kernel_port_as_iframe(18789, path="/", height="720")""")

md("""### 6 — (Fallback) chat from a notebook cell

If the embedded UI doesn't render in your browser, you can still chat here — same gateway,
same model. Edit the prompt and re-run.""")
code("""import subprocess
PATHX = 'export PATH="$(npm prefix -g)/bin:$PATH"; '
def chat(prompt):
    r = subprocess.run(["bash", "-c", PATHX +
        f"openclaw infer model run --gateway --model vllm/{MODEL} --prompt {prompt!r} --json"],
        capture_output=True, text=True)
    print(r.stdout or r.stderr)

chat("Say hello in one short sentence.")""")

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
