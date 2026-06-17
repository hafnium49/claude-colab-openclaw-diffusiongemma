#!/usr/bin/env python3
"""Generate notebooks/openclaw_diffusiongemma_colab.ipynb — the NOTEBOOK COUNTERPART of the
tested DiffusionGemma bash path.

  MASTER (source of truth):  bin/colab_openclaw_diffusiongemma.sh --gpu L4
                             --config configs/diffusiongemma_nvfp4.json  ->  drives
                             remote/remote_colab_openclaw_diffusiongemma.py on a Colab L4 from
                             OUTSIDE, headless, via the colab CLI (validated full e2e 2026-06-17).

  THIS NOTEBOOK (counterpart): the SAME bootstrap -> prompt phases, but as in-Colab cells you
                             Run-all on an L4 runtime, plus an interactive chat cell and the
                             inline OpenClaw dashboard. This is the "I want to chat with OpenClaw
                             in the Colab UI" path.

KEEP IN SYNC with the bash master + configs/diffusiongemma_nvfp4.json (model, ports, serve args,
OpenClaw compat fixes). Change the bash master FIRST, mirror here. .ipynb is built with json.dump
(always valid), never hand-edited.

Why a notebook avoids the keep-alive headache: the bash master drives a HEADLESS Colab VM, which
older `colab` CLIs (<=0.5.x) let Colab idle-prune at ~10-12 min (keep-alive 403; fixed in 0.6.0 via
a tunnel ping). A NOTEBOOK runs in a runtime your BROWSER owns — the open tab IS the heartbeat — so
there is no keep-alive RPC and no ~12-min cap. The long DiffusionGemma cold start just works while
the tab is open.
"""
import json, os

# Single source for the knobs the bash harness/config also use — change in lockstep.
MODEL_ID  = "RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"   # HF id == vLLM served-model id
MODEL_REF = "vllm/" + MODEL_ID                            # OpenClaw model ref (provider/model)
SHORT     = "DiffusionGemma-26B-A4B-NVFP4"
VLLM_PRE  = ("https://wheels.vllm.ai/nightly/cu129", "https://download.pytorch.org/whl/cu129")
LLM_PORT  = "8000"
GW_PORT   = "18789"

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(f"""# OpenClaw + DiffusionGemma (NVFP4) on a Colab L4 — chat in the notebook UI

Notebook counterpart of the validated bash master
`bin/colab_openclaw_diffusiongemma.sh --gpu L4 --config configs/diffusiongemma_nvfp4.json`
(which drives `remote/remote_colab_openclaw_diffusiongemma.py` on a Colab L4 headlessly via the
`colab` CLI). Here the **same bootstrap → prompt phases** run as in-cell steps on **your own L4
runtime**, so you can chat with OpenClaw right here.

**Setup:** `Runtime → Change runtime type → L4 GPU` → Save, then `Runtime → Run all`. **Keep this
tab open** — your browser is the runtime heartbeat (this is also why there's no ~10-min keep-alive
cap here, unlike the headless CLI path). First run installs vLLM + downloads the ~13 GB NVFP4
checkpoint + loads it (~10–12 min, one time).

> **L4 required.** DiffusionGemma-26B-A4B-NVFP4 needs ~24 GB VRAM (Colab Pro / compute units). It
> will **not** fit on a free T4.""")

md(f"""## 📋 Briefing — what & why

**Goal.** Stand up the tested DiffusionGemma stack — vLLM serving `{MODEL_ID}` on loopback, an
OpenClaw Gateway pointed at it — and chat with it from the notebook, with no public tunnel.

**Why vLLM here (not llama.cpp).** DiffusionGemma is a block-diffusion model shipped as **NVFP4**;
vLLM loads it on an L4 (Ada sm_89) via the **Marlin FP4 weight-only** fallback (NVFP4 is
Blackwell-native, but the weight-only path works on Ada). `DiffusionGemmaForBlockDiffusion` needs
`--trust-remote-code`. (The llama.cpp path in `openclaw_chat_colab.ipynb` is the fee-free T4 floor
model; this is the heavyweight target on an L4.)

**Containment.** Everything is loopback: vLLM on `127.0.0.1:{LLM_PORT}`, OpenClaw gateway on
`127.0.0.1:{GW_PORT}`. Nothing is exposed off the VM.

**Phase map (this notebook ⟷ the `bin/` master).**

| Notebook cell | Master phase (remote action) |
|---|---|
| 1 install | `bootstrap` → `install_vllm` + OpenClaw install |
| 2 serve | `bootstrap` → `start_vllm` (serve :{LLM_PORT}, NVFP4 on L4) |
| 3 onboard + gateway | `bootstrap` → `configure_openclaw` (compat fixes) + `start_openclaw_gateway` |
| 4 chat | `prompt` → `_prompt_run` infer |
| 5 dashboard | notebook-only (works because your browser owns the runtime) |

DiffusionGemma is a **reasoning** model (`enable_thinking` is on) — replies include a thinking
trace and take a while; that's expected.

---""")

md("### 1 — Install vLLM (nightly cu129) + OpenClaw  *(master: `bootstrap` → install)*")
code(f"""import subprocess, sys
print("nvidia-smi (confirm this is an L4 with ~24 GB) ...", flush=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True)

print("\\nInstalling vLLM (nightly cu129; ~3-5 min) ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "vllm", "--pre",
                "--extra-index-url", "{VLLM_PRE[0]}",
                "--extra-index-url", "{VLLM_PRE[1]}"], check=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "huggingface_hub", "hf_transfer"], check=True)

print("Installing OpenClaw (npm-based installer) ...", flush=True)
subprocess.run("curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard",
               shell=True, check=True)
print("\\nInstall complete.")""")

md(f"### 2 — Serve {SHORT} on :{LLM_PORT} + wait for ready  *(master: `bootstrap` → `start_vllm`)*")
code(f"""import subprocess, sys, os, time, glob, shutil, urllib.request
MODEL = "{MODEL_ID}"

# Put the CUDA runtime libs from the nvidia-* pip wheels on the loader path (mirrors the bash
# master's start_vllm) so `import vllm._C` finds libcudart.
ld = ":".join(glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib"))
env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1", VLLM_USE_V2_MODEL_RUNNER="1")
if ld:
    env["LD_LIBRARY_PATH"] = ld + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
# Optionally set HF_TOKEN in the cell above if the checkpoint is gated: env["HF_TOKEN"] = "hf_..."

# vLLM serve args == configs/diffusiongemma_nvfp4.json -> serve_args. The JSON-valued flags are
# passed as single list elements (NO shell), so they don't need the shlex-quoting the bash path did.
vllm = shutil.which("vllm") or "vllm"
serve = [vllm, "serve", MODEL,
    "--trust-remote-code", "--max-num-seqs", "1", "--gpu-memory-utilization", "0.90",
    "--max-model-len", "8192",
    "--hf-overrides", '{{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}}',
    "--default-chat-template-kwargs", '{{"enable_thinking":true}}',
    "--host", "127.0.0.1", "--port", "{LLM_PORT}"]
subprocess.Popen(serve, env=env,
                 stdout=open("/content/vllm.log", "w"), stderr=subprocess.STDOUT)

print("Downloading ~13 GB NVFP4 + loading on the L4 (~10-12 min, one time) ...", flush=True)
t0 = time.time(); ready = False
while time.time() - t0 < 1500:
    try:
        urllib.request.urlopen("http://127.0.0.1:{LLM_PORT}/v1/models", timeout=4); ready = True; break
    except Exception:
        print(f"   loading {{int(time.time()-t0)}}s ...", flush=True)
        subprocess.run("tail -1 /content/vllm.log", shell=True); time.sleep(10)
print(("vLLM READY after %ds" % (time.time()-t0)) if ready else
      "TIMEOUT — check /content/vllm.log (OOM? confirm this is a 24 GB L4)")""")

md(f"### 3 — Onboard OpenClaw against :{LLM_PORT} + start gateway  *(master: `bootstrap` → `configure_openclaw` + gateway)*")
code(f"""import subprocess, shutil, os, time
# Resolve openclaw by ABSOLUTE path -> never 'openclaw: command not found' (npm symlinks it into
# the global bin). Provider id 'vllm' matches the bash harness + the chat/dashboard cells.
OPENCLAW = shutil.which("openclaw") or "/usr/bin/openclaw"
os.environ.setdefault("OPENCLAW_GATEWAY_TOKEN", "dg-local-token")
def oc(args): return subprocess.run([OPENCLAW] + args, capture_output=True, text=True)
print("openclaw:", OPENCLAW)

ob = oc(["onboard", "--non-interactive", "--accept-risk", "--mode", "local",
    "--auth-choice", "custom-api-key", "--custom-provider-id", "vllm",
    "--custom-base-url", "http://127.0.0.1:{LLM_PORT}/v1",
    "--custom-model-id", "{MODEL_ID}", "--custom-compatibility", "openai",
    "--custom-api-key", "vllm-local", "--custom-text-input",
    "--gateway-port", "{GW_PORT}", "--gateway-bind", "loopback", "--gateway-auth", "token",
    "--gateway-token-ref-env", "OPENCLAW_GATEWAY_TOKEN",
    "--skip-daemon", "--skip-skills", "--skip-channels", "--skip-health", "--skip-ui", "--json"])
print("onboard rc =", ob.returncode)

# Infer fixes == configs/diffusiongemma_nvfp4.json -> openclaw.compat (string content; tools off;
# maxTokens < max-model-len so the completion can't overflow the window -> empty reply).
for k, v in [("compat.requiresStringContent", "true"), ("compat.supportsTools", "false"),
             ("maxTokens", "512"), ("contextWindow", "4096")]:
    oc(["config", "set", f"models.providers.vllm.models[0].{{k}}", v])

# Gateway runs in-process (lives while the tab is open) — needed for the inline dashboard (cell 5).
subprocess.Popen([OPENCLAW, "gateway", "run"],
    stdout=open("/content/gateway.log", "w"), stderr=subprocess.STDOUT)
time.sleep(12)
print("gateway started on 127.0.0.1:{GW_PORT}")""")

md(f"""### 4 — 💬 Chat with DiffusionGemma  *(master: `prompt` → `_prompt_run`)*

Edit `MESSAGE` and re-run this cell per turn. Uses **direct** infer (no gateway needed, the robust
path). DiffusionGemma reasons before answering, so a reply takes a while and includes a thinking
trace — that's the model, not a hang.""")
code(f'''MESSAGE   = "Hello! In one sentence, who are you?"   # <- edit me; re-run for each turn
MODEL_REF = "{MODEL_REF}"

import subprocess, json, shutil
OPENCLAW = shutil.which("openclaw") or "/usr/bin/openclaw"
r = subprocess.run([OPENCLAW, "infer", "model", "run", "--model", MODEL_REF,
                    "--prompt", MESSAGE, "--json"], capture_output=True, text=True)
try:
    print(json.loads(r.stdout)["outputs"][0]["text"])
except Exception:
    print(r.stdout or r.stderr)''')

md(f"""### 5 — (Optional) OpenClaw Control dashboard, inline — no tunnel

Works **only because your browser owns this runtime**, so `serve_kernel_port_as_iframe` mints a
Google-authenticated proxy to the gateway port {GW_PORT} — no public tunnel. (This is exactly why
the dashboard is NOT reachable when the headless CLI manages the VM.) Needs the gateway from cell 3.""")
code(f'''from google.colab import output
import os
print("If the dashboard asks for a token, paste:", os.environ.get("OPENCLAW_GATEWAY_TOKEN", "dg-local-token"))
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
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openclaw_diffusiongemma_colab.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
