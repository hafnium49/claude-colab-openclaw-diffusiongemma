#!/usr/bin/env python3
from __future__ import annotations

import ast
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
    'configs/diffusiongemma_web.json',
    'configs/diffusiongemma_research.json',
    'configs/smoke_test_tiny.json',
    'configs/llama_qwen9b.json',
    'configs/llama_smoke.json',
    'configs/llama_lfm2.json',
    'configs/lfm2_ollama_web.json',
    'configs/lfm2_ollama_research.json',
    'configs/colab_ai_gemini.json',
    'examples/prompt_task.json',
    'examples/research_task.json',
    'examples/web_verify_task.json',
    'examples/web_research_deep.json',
    'examples/web_research_fanout.json',
    'notebooks/_gen_notebook.py',
    'notebooks/openclaw_chat_colab.ipynb',
    'notebooks/_gen_colab_ai_notebook.py',
    'notebooks/openclaw_colab_ai.ipynb',
    'notebooks/_gen_diffusiongemma_notebook.py',
    'notebooks/openclaw_diffusiongemma_colab.ipynb',
]

for rel in required:
    path = ROOT / rel
    if not path.exists():
        raise SystemExit(f'Missing required file: {rel}')

for rel in ['configs/diffusiongemma_nvfp4.json', 'configs/diffusiongemma_web.json',
            'configs/diffusiongemma_research.json',
            'configs/smoke_test_tiny.json',
            'configs/llama_qwen9b.json',
            'configs/llama_smoke.json', 'configs/llama_lfm2.json',
            'configs/lfm2_ollama_web.json', 'configs/lfm2_ollama_research.json',
            'configs/colab_ai_gemini.json',
            'examples/prompt_task.json', 'examples/research_task.json', 'examples/web_verify_task.json',
            'examples/web_research_deep.json', 'examples/web_research_fanout.json',
            'notebooks/openclaw_chat_colab.ipynb', 'notebooks/openclaw_colab_ai.ipynb',
            'notebooks/openclaw_diffusiongemma_colab.ipynb']:
    with (ROOT / rel).open('r', encoding='utf-8') as f:
        json.load(f)

for rel in ['remote/remote_colab_openclaw_diffusiongemma.py', 'remote/colab_exec_stub.py',
            'notebooks/_gen_notebook.py', 'notebooks/_gen_colab_ai_notebook.py',
            'notebooks/_gen_diffusiongemma_notebook.py']:
    py_compile.compile(str(ROOT / rel), doraise=True)

# Every notebook code cell must be valid Python — the real failure mode for templated .ipynb
# (a broken f-string/escape) that plain JSON-parsing above would not catch.
for rel in ['notebooks/openclaw_chat_colab.ipynb', 'notebooks/openclaw_colab_ai.ipynb',
            'notebooks/openclaw_diffusiongemma_colab.ipynb']:
    nb = json.loads((ROOT / rel).read_text(encoding='utf-8'))
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = cell['source']
        src = ''.join(src) if isinstance(src, list) else src
        try:
            ast.parse(src)
        except SyntaxError as exc:
            raise SystemExit(f'{rel} code cell {i} is not valid Python: {exc}')

subprocess.run(['bash', '-n', str(ROOT / 'bin/colab_openclaw_diffusiongemma.sh')], check=True)
print('self_test_ok')
