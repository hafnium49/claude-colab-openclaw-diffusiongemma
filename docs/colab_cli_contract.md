# Colab CLI contract

The launcher expects the following Colab CLI commands to be available locally:

```text
colab new -s NAME --gpu GPU
colab status -s NAME
colab upload -s NAME LOCAL REMOTE
colab exec -s NAME -f FILE
colab download -s NAME REMOTE LOCAL
colab log -s NAME -o FILE
colab stop -s NAME
```

The local script uploads only the remote orchestrator, the config JSON, and the task JSON. Then it runs a small stub that executes `/content/remote_colab_openclaw_diffusiongemma.py` inside the Colab kernel.

The remote script uses control files:

```text
/content/ocdg_config.json
/content/ocdg_task.json
/content/ocdg_control.json
```

The remote result path is:

```text
/content/openclaw_diffusiongemma_results.zip
```
