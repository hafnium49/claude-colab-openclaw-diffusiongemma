#!/usr/bin/env python3
from __future__ import annotations

import json
import py_compile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

required = [
    'README.md',
    '.claude/agents/colab-openclaw-diffusiongemma.md',
    '.claude/skills/colab-openclaw-diffusiongemma/SKILL.md',
    'bin/colab_openclaw_diffusiongemma.sh',
    'remote/remote_colab_openclaw_diffusiongemma.py',
    'remote/colab_exec_stub.py',
    'configs/diffusiongemma_nvfp4.json',
    'configs/smoke_test_tiny.json',
    'configs/llama_qwen9b.json',
    'configs/llama_smoke.json',
    'configs/colab_ai_gemini.json',
    'examples/prompt_task.json',
    'examples/research_task.json',
    'notebooks/_gen_notebook.py',
    'notebooks/openclaw_chat_colab.ipynb',
]

for rel in required:
    path = ROOT / rel
    if not path.exists():
        raise SystemExit(f'Missing required file: {rel}')

for rel in ['configs/diffusiongemma_nvfp4.json', 'configs/smoke_test_tiny.json',
            'configs/llama_qwen9b.json', 'configs/llama_smoke.json', 'configs/colab_ai_gemini.json',
            'examples/prompt_task.json', 'examples/research_task.json',
            'notebooks/openclaw_chat_colab.ipynb']:
    with (ROOT / rel).open('r', encoding='utf-8') as f:
        json.load(f)

for rel in ['remote/remote_colab_openclaw_diffusiongemma.py', 'remote/colab_exec_stub.py',
            'notebooks/_gen_notebook.py']:
    py_compile.compile(str(ROOT / rel), doraise=True)

subprocess.run(['bash', '-n', str(ROOT / 'bin/colab_openclaw_diffusiongemma.sh')], check=True)
print('self_test_ok')
