# Ntweak

**Steer neurons. Chat with the difference. Run it locally.**

Ntweak is a local neuron steering playground built around **Qwen3.5 4B**. Download a model, adjust individual MLP neuron activations, and explore how its responses change. Import community recipes from [Neuronary](https://neuronary.org), create your own, and compare steered responses against an unsteered baseline.

The app runs a Python server and a browser UI on your computer. It uses local model weights for inference, with no hosted inference service, API subscription, or Node build step.

## Features

- **Live steering rack:** combine neurons across layers, adjust signed doses, bypass individual tweaks, or switch all steering off.
- **Local streaming chat:** chat with the model while interventions are applied to every prompt and generated token.
- **Baseline comparison:** run the same input with and without steering, with separate conversation histories after branching.
- **Recipe sharing:** import a Neuronary link or JSON, save recipes locally, and export recipes and conversations as JSON.
- **Model library:** inspect and download files from Hugging Face model repositories, resume interrupted transfers, reuse cached checkpoints, or add an existing local folder.
- **Visible intervention details:** inspect the measured residual median, weight-column norm, and activation delta used for each response.
- **Local acceleration:** choose Apple Metal (MPS), NVIDIA CUDA, or CPU.

## Quick start

Requirements: **Python 3.11+**, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/). Internet access is needed to install dependencies and download a model; subsequent local chat can run offline.

```sh
git clone https://github.com/arun-seran/ntweak.git
cd ntweak
uv sync --locked
uv run python -m ntweak --open
```

The app opens at **http://127.0.0.1:8787**. Keep the terminal running while you use it; press **Ctrl+C** to stop the server.

On **macOS**, you can instead double-click **Start Ntweak.command** after cloning. The launcher installs dependencies into this project's `.venv` and opens the app. If uv is unavailable, it uses Python's `venv` and pip.

### Download and load Qwen3.5 4B

1. Open **Models** and select **Get this model** on the Qwen3.5 4B card.
2. Review the selected files and choose **Download selected**.
3. When the download finishes, choose **Load model**. Leave the device on **Auto** or select an available backend.
4. Import a recipe or add neuron tweaks in the steering rack, then send a message.
5. Enable **Compare with baseline** to see both branches. Use **New chat** when starting a fresh experiment.

Original bfloat16 weights require roughly **8–9 GB of disk space**, plus working memory. MPS/CUDA use bfloat16; CPU uses float32 and needs more memory. **16 GB RAM or more** is a practical starting point, but available memory, backend, and conversation length determine what fits. Defaults reserve a 4,096-token context budget, including up to 512 output tokens.

### Install without uv

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m ntweak --open
```

On Windows, use `py -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell instead of the first two commands. CPU and CUDA availability depend on your PyTorch installation and hardware. macOS uses MPS when available.

The UI and downloader can run with Python's standard library alone using `python3 -m ntweak`. Inference requires the installed project dependencies: PyTorch 2.6+, Transformers 5.2+, and the other packages in `pyproject.toml`.

## Use your recipe

Choose **Import recipe**, paste a Neuronary link such as [this example](https://neuronary.org/r/9j43ped5), and import. The app reads the public API used by [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer/blob/main/explorer/agent_mcp.py). Imports are read-only; the app does not publish recipes to Neuronary. You can also use the **Paste JSON** tab:

```json
{
  "name": "My experiment",
  "model": "Qwen/Qwen3.5-4B",
  "neurons": [
    {"layer": 22, "neuron": 123, "dose": 0.8, "mode": "add"}
  ],
  "sampling": {
    "temperature": 0.7,
    "min_p": 0.05,
    "repetition_penalty": 1.1,
    "seed": 42
  }
}
```

The indices above demonstrate the format; they are **not** the contents of the linked recipe and have no claimed semantic effect. All neuron and layer indices are zero-based. In Qwen3.5 4B, layers range from **0–31** and MLP neurons from **0–9,215**.

`steer_neurons`, `n`, per-neuron `enabled`, and multiplier mode (`"mode": "mul"`) are also accepted. Positive additive doses push along a neuron's down-projection direction; negative doses push against it. Effects are experimental and depend on the model, neuron, dose, and prompt.

Saved recipes and conversations stay in this browser's local storage. Export recipes or chats as JSON for a durable backup. Starting a new chat clears the current conversation from the workspace.

## Exact intervention

For additive steering, on **each user input**:

1. Render the full conversation with the model's own chat template.
2. Run one **unsteered** forward pass through the language backbone.
3. At each targeted decoder layer `L`, capture the residual stream **leaving the layer**, take its L2 norm per token, exclude token 0 and padding, then take the median.
4. Compute `delta = dose × median(‖resid_L[:, 1:, :]‖₂) / ‖W_down[:, n]‖₂` in float32.
5. Install a forward **pre-hook** on `layer.mlp.down_proj`. Add `delta` to neuron `n` in its input, the `silu(gate) * up` activations. Apply it on **every prompt token and every generated token**. Multiple neurons/layers compose in the same run.
6. Remove all hooks after completion, stop, or error. The checkpoint is never modified.

Multiplier mode instead multiplies that activation by `dose`; 0 ablates the neuron and 1 leaves it unchanged. Duplicate rows are applied in recipe order within a layer. The response's **View applied doses** shows the actual residual medians, column norms, and activation deltas.

**Calibration differs from upstream:** [neuron-explorer's runtime](https://github.com/lyramakesmusic/neuron-explorer/blob/main/explorer/runtime.py) uses cached corpus statistics or a fixed reference text. Ntweak measures the **current conversation afresh for each input**. The formula and intervention site match, but a published recipe may produce different text because its calibration, prompt, dtype, backend, and sampling differ. Exact published-output reproduction is not guaranteed. The app disables thinking by default; enable it in generation settings if desired.

## Other models

Enter any Hugging Face **model repository ID and revision** in the model library, inspect the files, and select which ones to download. The recommended selection prefers safetensors over duplicate PyTorch `.bin` weights. Revisions are pinned to the inspected commit, downloads show byte progress, interrupted transfers resume, and downloaded files are checked against the repository's SHA256 or Git blob digest when supplied.

Existing complete Hugging Face cache snapshots are detected automatically. You can also add a local model directory. Loading and generation use `local_files_only=True` and `trust_remote_code=False`.

The inference runtime supports Transformers causal language models and the Qwen3.5 multimodal wrapper in **text-only chat**. Neuron steering requires accessible **dense floating-point `layers[L].mlp.down_proj` weights** (Qwen3.5 4B, Qwen3, common Llama/Mistral architectures). Other architectures may support baseline chat only. Downloading a repository does not guarantee inference compatibility. GGUF, MLX, packed quantization, custom remote model code, vision chat, and mixture-of-experts steering are not supported. Neuron identities are specific to a checkpoint: change the recipe's model field explicitly before applying it to a different model.

Gated/private repositories can use a read token in the UI or `HF_TOKEN` in the launch environment. Accept repository terms on Hugging Face first. Tokens are never persisted in the workspace, model registry, or job status, and are stripped on CDN redirects.

## Local storage and networking

- The server binds only to `127.0.0.1`. Host, Origin, and Fetch Metadata checks reject other-site API requests; a Content Security Policy blocks external UI resources.
- Models, download metadata, and the local folder registry live in `.data/` inside the project. Set `NTWEAK_DATA_DIR` or use `--data-dir /path/to/data` to change it.
- The UI has no CDN assets, analytics, or remote inference. Model loading has Hugging Face offline mode enabled. **Once dependencies and weights are downloaded, chat works offline.**
- Only explicit model inspections/downloads contact Hugging Face; only explicit recipe-link imports contact Neuronary's public API. JSON recipe import stays local.
- Browser storage is tied to the browser and the address/port. Keep using the same address; export your work before clearing browser data.
- **Stop** cancels between generated tokens (and between calibration layer hooks). A long GPU kernel or an in-flight download read must finish before cancellation takes effect. Baseline comparison runs sequentially to reduce peak memory and keeps separate assistant histories after branching.

## Configuration

```sh
uv run python -m ntweak --port 8788 --data-dir /path/to/local-model-data --open
```

- `--port`: defaults to `8787`.
- `--data-dir`: defaults to `.data/` in the project; `NTWEAK_DATA_DIR` can also set it.
- `--open`: opens the app in your default browser after the server starts.
- `HF_TOKEN`: optional read token for gated/private Hugging Face repositories.

The bind address stays on loopback; there is no public hosting mode.

## Development and tests

The application is intentionally small: a standard-library HTTP server, a local Transformers runtime, and plain HTML/CSS/JavaScript.

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
```

The stdlib tests exercise HTTP parsing and responses through in-memory sockets, recipe validation, download resume/integrity/cancellation with in-memory transfer responses, and model library behavior. Tensor tests additionally run when PyTorch is installed, verifying the actual calibration and down-projection intervention. When Transformers is installed, integration tests construct a tiny random Llama model locally and run real generation through the chat runtime. A full 4B generation requires the downloaded checkpoint and sufficient hardware; no simulated responses are used by the app.

Optional DOM interaction tests cover model download/load controls, recipe import, streamed UTF-8 responses, separate comparison histories, local persistence, and safe rendering of imported text:

```sh
npm install
npm test
```

Node is used only for these development checks; the app itself has no Node dependency. These DOM tests do not replace a visual browser check.

Source layout:

```text
ntweak/
  server.py       Loopback HTTP API, background jobs, and chat streaming
  engine.py       Model loading, residual calibration, and neuron hooks
  models.py       Model inspection, local registry, and resumable downloads
  recipes.py      Recipe validation and Neuronary imports
  static/         Browser UI; no frontend build required
tests/            HTTP, download, recipe, tensor, generation, and DOM checks
```

## Troubleshooting

- **Missing inference runtime:** run `uv sync`, then restart with `uv run python -m ntweak`. A plain system Python may not see the project's dependencies.
- **Memory error:** unload other models, reduce the context/output limits, and use one chat branch. Full precision original weights are required by this implementation; downloading a smaller compatible dense model is another option.
- **Unsupported model configuration:** use a current Transformers 5.x release (`uv sync --upgrade`), original safetensors weights, and no custom remote model code.
- **Recipe unavailable:** check the link and connection, or paste the recipe's exported JSON in the import dialog. Recipes are fetched when imported rather than bundled with the app.
- **Recipe/model mismatch:** load the model named by the recipe. Neuron indices do not imply the same behavior across checkpoints.
- **Port already in use:** choose another port, for example `uv run python -m ntweak --port 8788 --open`.
- **Conversations appear missing:** use the same browser and address/port as before. Browser storage is separate for `localhost`, `127.0.0.1`, and different ports.

## Contributing

Bug reports, reproducible steering experiments, and pull requests are welcome at [arun-seran/ntweak](https://github.com/arun-seran/ntweak). For inference issues, include the model ID/revision, OS, backend, dependency versions, and a small recipe and prompt that reproduce the problem. Omit access tokens and private conversations.

Run the Python and DOM checks before submitting changes. Please keep downloaded weights, local caches, environments, and personal workspace data out of commits; `.gitignore` excludes the default locations.

## License

Ntweak is licensed under the [MIT License](LICENSE). Model weights are downloaded separately and remain subject to their respective licenses and access terms.

## Acknowledgements

The intervention approach is informed by [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), an MIT-licensed neuron exploration project. Community recipes come from [Neuronary](https://neuronary.org). Model inference uses [Qwen's Transformers implementation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5) and [PyTorch](https://pytorch.org).
