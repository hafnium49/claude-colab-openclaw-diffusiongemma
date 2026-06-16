#!/usr/bin/env python3
"""Generate notebooks/openclaw_colab_ai.ipynb — PATH A: OpenClaw + FREE google.colab.ai (Gemini).

  MASTER (source of truth):  bin/colab_openclaw_diffusiongemma.sh --config configs/colab_ai_gemini.json
                             -> remote/remote_colab_openclaw_diffusiongemma.py, serve.backend "colab_ai".
                             That backend CANNOT actually run headless (see below), so this notebook is
                             the real entry point for Path A; the master config exists only for parity.

  THIS NOTEBOOK: the same phases (shim -> onboard -> prompt -> multi-step task), but as in-Colab
                 cells on a CPU runtime, because google.colab.ai is BROWSER-UI-ONLY.

WHY A NOTEBOOK, NOT THE CLI. `from google.colab import ai` can only fetch its MODEL_PROXY_API_KEY
from the Colab UI; a headless `colab exec` raises
`TimeoutException: ... only fetched from the Colab UI`. So Path A must run in a browser tab. For the
same reason the shim here runs IN THIS KERNEL PROCESS (a background uvicorn thread) and we PRIME the
proxy key on the main thread first (cell 1) — a detached subprocess (as in the master shim) would
fail to fetch the key. This is the one place this notebook deliberately differs from the master.

TRADE-OFF. Inference runs on Google's backend, so prompts LEAVE the VM — fee-free but NOT
sandbox-contained, unlike the llama.cpp / vLLM self-hosted paths (the master's default).

Authoring .ipynb JSON by hand is error-prone, so this builds it with json.dump (always valid).
Code cells are raw strings (r'''...''') so `\n` / f-string braces survive verbatim into the notebook.
"""
import json, os

# Knobs (mirror configs/colab_ai_gemini.json). AI_MODEL default is what the user asked for; the
# notebook's cell 1 prints the live ai.list_models() catalog and warns if it isn't offered.
AI_MODEL_DEFAULT = "google/gemini-3.5-flash"
FALLBACK_HINT = "google/gemini-2.5-flash"
PORT = 8000
GW_PORT = 18789
MAX_TOKENS = 2048
CONTEXT_WINDOW = 8192

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(f"""# OpenClaw + FREE `google.colab.ai` (Gemini) on Colab — Path A (browser notebook)

**Free, no GPU, no API key — but BROWSER-ONLY and NOT sandbox-contained.** This is the notebook
counterpart of `serve.backend: "colab_ai"` in the bash master
(`bin/colab_openclaw_diffusiongemma.sh --config configs/colab_ai_gemini.json`). Unlike the
self-hosted paths (llama.cpp / vLLM), inference runs on **Google's** backend via `google.colab.ai`,
so your prompts leave the VM. It is fee-free and needs only a **CPU** runtime, but `google.colab.ai`
can only fetch its proxy key from the **Colab UI**, so this CANNOT be driven by the headless `colab`
CLI — run it here, in a browser tab.

**Setup:** `Runtime → Change runtime type → CPU` (no GPU needed) → Save, then `Runtime → Run all`.
Keep the tab open (your browser is the runtime heartbeat). Default model is `{AI_MODEL_DEFAULT}` —
cell 1 verifies it against the live catalog.""")

md(f"""## 📋 Briefing — what & why

**Goal.** Run OpenClaw with a **free** LLM and no API key. The free LLM is `google.colab.ai`
(Gemini / Gemma, included with Colab). OpenClaw speaks OpenAI HTTP but `google.colab.ai` is a Python
function — so a tiny FastAPI **shim** exposes it as `http://127.0.0.1:{PORT}/v1`, and OpenClaw
onboards against the shim exactly like any other backend.

**⚠️ Trade-off vs. the self-hosted paths.** This is **NOT contained**: `google.colab.ai` calls
Google's servers, so prompts leave the sandbox. The llama.cpp / vLLM paths (the master's default)
keep the model **on the VM** — fee-free *and* contained. Use Path A only when leaving the sandbox is
acceptable.

**⚠️ Browser-only, hence in-kernel shim.** `google.colab.ai` raises
`MODEL_PROXY_API_KEY ... only fetched from the Colab UI` from a headless session. So the shim runs
**in this kernel process** (a background uvicorn thread), NOT a detached subprocess, and **cell 1
primes the proxy key on the main thread** so the shim thread reuses the cached key. That is the one
place this notebook deliberately differs from the master's `colab_ai` shim (which can't run headless
anyway). If cell 1's prime fails, the shim will return `[colab_ai error]` — fix cell 1 first.

**Phase map (this notebook ⟷ the `bin/` master, `--config configs/colab_ai_gemini.json`).**

| Notebook cell | Master phase (remote action) |
|---|---|
| 1 verify + pick model | `ai.list_models()` guard + prime key (no GPU, no download) |
| 2 shim + serve | `bootstrap` → `install_colab_ai` + `start_colab_ai` (shim on :{PORT}) |
| 3 onboard + gateway | `bootstrap` → `configure_openclaw` (compat fixes) + gateway |
| 4 chat (smoke) | `prompt` → `_prompt_run` infer |
| 5 autonomous task | `task` → `_task_run` (multi-step `steps[]` → `research_result.md`) |
| 6 dashboard | notebook-only (your browser owns the runtime) |

---""")

md(f"""### 1 — Verify `google.colab.ai` + pick model  *(no GPU, no download)*

Default `AI_MODEL` is `{AI_MODEL_DEFAULT}` (as requested). This prints the **live**
`ai.list_models()` catalog and warns if that model isn't offered — if so, set `AI_MODEL` to one it
lists (e.g. `{FALLBACK_HINT}`) and re-run. It also **primes** the proxy key in this kernel so the
shim thread (cell 2) can reuse it.""")
code(r'''AI_MODEL = "google/gemini-3.5-flash"   # <- EDIT ME. colab.ai model the shim will call.
PORT     = 8000
OC_MODEL_ID = AI_MODEL.split("/", 1)[-1]   # what OpenClaw sees -> infer with vllm/<this>

from google.colab import ai
available = []
try:
    available = list(ai.list_models())
except Exception as e:
    print("google.colab.ai unavailable — are you on a Colab BROWSER runtime?:", e)

print("Live colab.ai catalog:")
for m in available:
    print("  -", m)

if available and AI_MODEL not in available:
    print(f"\n⚠️  {AI_MODEL!r} is NOT in the live catalog above.")
    print("    Set AI_MODEL to one that IS listed (e.g. 'google/gemini-2.5-flash'), then re-run.")
else:
    print(f"\n✓ AI_MODEL = {AI_MODEL}")

# Prime MODEL_PROXY_API_KEY in THIS (main) kernel thread so the shim's worker thread reuses it.
_prime = AI_MODEL if AI_MODEL in available else (available[0] if available else AI_MODEL)
try:
    ai.generate_text("ping", model_name=_prime)
    print(f"✓ colab.ai reachable (proxy key primed via {_prime})")
except Exception as e:
    print("✗ colab.ai generate_text failed — fix this before running cell 2:", e)''')

md(f"""### 2 — Install OpenClaw + FastAPI shim, serve `google.colab.ai` on :{PORT}  *(master: `bootstrap` → `install_colab_ai` + `start_colab_ai`)*

The shim runs in a **background thread of this kernel** (see briefing) so its `ai.generate_text`
calls keep the primed UI proxy key. OpenClaw (a separate process) only ever talks to the shim over
loopback — it never imports `google.colab.ai` itself.""")
code(r'''import subprocess, sys, time, threading, json, urllib.request

print("Installing fastapi + uvicorn ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "fastapi", "uvicorn"], check=True)
print("Installing OpenClaw (npm-based installer) ...", flush=True)
subprocess.run("curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard",
               shell=True, check=True)

# OpenAI-compatible shim over google.colab.ai — defined AND served IN THIS KERNEL PROCESS so the
# proxy key primed in cell 1 stays usable (a detached subprocess could not fetch it).
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from google.colab import ai
import uvicorn

app = FastAPI()

@app.get("/v1/models")
def _models():
    try:    ids = list(ai.list_models())
    except Exception:  ids = [AI_MODEL]
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "google"} for m in ids]}

def _content(m):
    c = m.get("content", "")
    return c if isinstance(c, str) else json.dumps(c)

@app.post("/v1/chat/completions")
async def _chat(req: Request):
    body = await req.json()
    msgs = body.get("messages", [])
    prompt = "\n\n".join(f"{m.get('role','user')}: {_content(m)}" for m in msgs) or "Hello"
    try:    text = ai.generate_text(prompt, model_name=AI_MODEL)   # text-to-text, non-streaming
    except Exception as e:  text = "[colab_ai error] " + repr(e)
    created, model = int(time.time()), (body.get("model") or AI_MODEL)
    # OpenClaw's `infer` requests stream:true -> we MUST answer with Server-Sent Events (one full
    # delta chunk + a finish chunk + [DONE]); a plain JSON body makes OpenClaw report
    # "Stream ended without finish_reason" and drop the text. (llama.cpp works because it streams.)
    if body.get("stream"):
        def _sse():
            head = {"id": "chatcmpl-colabai", "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [{"index": 0,
                    "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
            tail = {"id": "chatcmpl-colabai", "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield "data: " + json.dumps(head) + "\n\n"
            yield "data: " + json.dumps(tail) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_sse(), media_type="text/event-stream")
    return {"id": "chatcmpl-colabai", "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}], "usage": {}}

# uvicorn in a daemon THREAD (not a subprocess, not the main loop) -> shares this process + proxy key.
_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
threading.Thread(target=_server.run, daemon=True).start()

ok = False
for _ in range(60):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read(); ok = True; break
    except Exception:
        time.sleep(1)
print(f"shim {'serving' if ok else 'DID NOT COME UP for'} google.colab.ai on http://127.0.0.1:{PORT}/v1")''')

md(f"""### 3 — Onboard OpenClaw against :{PORT} + start gateway  *(master: `bootstrap` → `configure_openclaw` + gateway)*

Provider id is kept `vllm` so the model ref (`vllm/<id>`) matches the bash harness and the chat /
task cells. The compat fixes mirror `configs/colab_ai_gemini.json` (string content; tools off;
`maxTokens` {MAX_TOKENS} < `contextWindow` {CONTEXT_WINDOW}).""")
code(r'''import subprocess, shutil, os, time
# Resolve openclaw by ABSOLUTE path -> never 'openclaw: command not found' (npm symlinks into /usr/bin).
OPENCLAW = shutil.which("openclaw") or "/usr/bin/openclaw"
os.environ.setdefault("OPENCLAW_GATEWAY_TOKEN", "colabai-local-token")
def oc(args): return subprocess.run([OPENCLAW] + args, capture_output=True, text=True)
print("openclaw:", OPENCLAW, "| model id:", OC_MODEL_ID)

ob = oc(["onboard", "--non-interactive", "--accept-risk", "--mode", "local",
    "--auth-choice", "custom-api-key", "--custom-provider-id", "vllm",
    "--custom-base-url", f"http://127.0.0.1:{PORT}/v1",
    "--custom-model-id", OC_MODEL_ID, "--custom-compatibility", "openai",
    "--custom-api-key", "colabai-local", "--custom-text-input",
    "--gateway-port", "18789", "--gateway-bind", "loopback", "--gateway-auth", "token",
    "--gateway-token-ref-env", "OPENCLAW_GATEWAY_TOKEN",
    "--skip-daemon", "--skip-skills", "--skip-channels", "--skip-health", "--skip-ui", "--json"])
print("onboard rc =", ob.returncode)

# Compat infer fixes (mirror configs/colab_ai_gemini.json): string content; tools off;
# maxTokens < contextWindow -> avoid empty/overflow completions.
for k, v in [("compat.requiresStringContent", "true"), ("compat.supportsTools", "false"),
             ("maxTokens", "2048"), ("contextWindow", "8192")]:
    oc(["config", "set", f"models.providers.vllm.models[0].{k}", v])

# Gateway runs in-process (lives while the tab is open) — needed only for the inline dashboard.
subprocess.Popen([OPENCLAW, "gateway", "run"],
    stdout=open("/content/gateway.log", "w"), stderr=subprocess.STDOUT)
time.sleep(12)
print("gateway started on 127.0.0.1:18789")''')

md(f"""### 4 — 💬 Chat (smoke test)  *(master: `prompt` → `_prompt_run`)*

Self-contained: resolves `openclaw` by absolute path and uses **direct** infer (no gateway needed).
Edit `MESSAGE`, re-run per turn. Each turn is one free `google.colab.ai` call via the shim.""")
code(r'''MESSAGE = "Hello! Who are you, in one sentence?"   # <- edit me; re-run for each turn

import subprocess, json, shutil
OPENCLAW = shutil.which("openclaw") or "/usr/bin/openclaw"
r = subprocess.run([OPENCLAW, "infer", "model", "run", "--model", f"vllm/{OC_MODEL_ID}",
                    "--prompt", MESSAGE, "--json"], capture_output=True, text=True)
try:
    print(json.loads(r.stdout)["outputs"][0]["text"])
except Exception:
    print(r.stdout or r.stderr)''')

md(f"""### 5 — 🤖 Autonomous task  *(master: `task` → `_task_run`)* — multi-step, no human

Mirrors the master's `task` worker: runs a **list of `STEPS`** sequentially through OpenClaw (one
free `google.colab.ai` call each) and accumulates the answers into `/content/research_result.md`,
exactly like `bin/colab_openclaw_diffusiongemma.sh --task examples/research_task.json`. Edit `TOPIC`
/ `STEPS`. The `infer()` salvage rule matches the master's `extract_infer_text` (only a non-empty
`outputs[0].text` counts).""")
code(r'''import subprocess, json, shutil
OPENCLAW  = shutil.which("openclaw") or "/usr/bin/openclaw"
TRANSPORT = "local"   # direct infer (no --gateway) — the robust path the master uses for research

TOPIC = "Practical trade-offs of a free hosted LLM (google.colab.ai) vs. self-hosting one on a GPU."
STEPS = [
    "List the dimensions to compare a free hosted LLM against a self-hosted local model (cost, privacy/containment, latency, rate limits, capability, offline use). One line each.",
    "For each dimension, say which side wins and why, in one sentence.",
    "Give 3 concrete situations where the free hosted option is the right call, and 3 where self-hosting is.",
    "Synthesize the above into a 5-bullet executive summary with a final recommendation heuristic.",
]

def infer(prompt):
    flag = ["--gateway"] if TRANSPORT == "gateway" else []   # else: direct infer, the robust path
    r = subprocess.run([OPENCLAW, "infer", "model", "run", *flag, "--model", f"vllm/{OC_MODEL_ID}",
                        "--prompt", prompt, "--json"], capture_output=True, text=True)
    try:    # same salvage rule as the master's extract_infer_text: only non-empty text counts
        outs = (json.loads(r.stdout) or {}).get("outputs") or []
        if outs and isinstance(outs[0], dict) and isinstance(outs[0].get("text"), str) and outs[0]["text"].strip():
            return outs[0]["text"]
    except Exception:
        pass
    return None

lines = [f"# Autonomous research result\n\n- Topic: {TOPIC}\n- Model: vllm/{OC_MODEL_ID}\n"]
for i, step in enumerate(STEPS, 1):
    print(f"step {i}/{len(STEPS)}: {step[:60]}...", flush=True)
    text = infer(step)
    lines.append(f"\n## Step {i}\n\n**Prompt:** {step}\n\n{text or '(no text returned)'}\n")
open("/content/research_result.md", "w").write("\n".join(lines))
print("\nWrote /content/research_result.md (" + str(len(STEPS)) + " steps)\n" + "=" * 60)
print(open("/content/research_result.md").read())''')

md(f"""### 6 — (Optional) OpenClaw Control dashboard, inline — no tunnel

Works **only because your browser owns this runtime**: `serve_kernel_port_as_iframe` mints a
Google-authenticated proxy to the gateway's port {GW_PORT} — no public tunnel. Requires the gateway
from cell 3.""")
code(r'''from google.colab import output
import os
print("If the dashboard asks for a token, paste:", os.environ.get("OPENCLAW_GATEWAY_TOKEN", "colabai-local-token"))
output.serve_kernel_port_as_iframe(18789, path="/", height="720")''')

nb = {
    "cells": cells,
    "metadata": {
        # CPU runtime (no accelerator key) — Path A needs no GPU.
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openclaw_colab_ai.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
