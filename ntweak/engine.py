"""Local Transformers runtime. PyTorch is imported only when inference is requested."""
from __future__ import annotations

from contextlib import contextmanager
import gc
import importlib.metadata
import importlib.util
import inspect
import threading
import time

from .models import Cancelled
from .recipes import integer, normalize_recipe


def dependencies():
    versions = {}
    for module in ("torch", "transformers", "safetensors"):
        try:
            versions[module] = importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError:
            versions[module] = None
    ready = all(versions.values())
    # Qwen3.5 support landed in Transformers 5.2.
    if versions["transformers"]:
        parts = versions["transformers"].split(".")
        try:
            ready = ready and (int(parts[0]), int(parts[1])) >= (5, 2)
        except ValueError:
            ready = False
    return {"ready": ready, "versions": versions,
            "install": "uv sync"}


def find_backbone(model):
    """Resolve Qwen3.5's multimodal wrapper and common dense causal LMs."""
    for path in ("model.language_model", "language_model.model", "model", "transformer", ""):
        obj = model
        for name in path.split(".") if path else []:
            obj = getattr(obj, name, None)
        if obj is not None and hasattr(obj, "layers"):
            return obj
    return None


def validate_targets(layers, neurons):
    for row in neurons:
        if not 0 <= row["layer"] < len(layers):
            raise ValueError(f"Layer {row['layer']} is outside this model (0–{len(layers)-1}).")
        mlp = getattr(layers[row["layer"]], "mlp", None)
        proj = getattr(mlp, "down_proj", None)
        if proj is None or not hasattr(proj, "weight") or proj.weight.ndim != 2:
            raise ValueError(f"Layer {row['layer']} has no dense mlp.down_proj. This architecture cannot use these neuron tweaks.")
        if not proj.weight.is_floating_point():
            raise ValueError("Packed/quantized projection weights are not supported for steering. Use original Transformers safetensors weights.")
        if not 0 <= row["neuron"] < proj.weight.shape[1]:
            raise ValueError(f"Neuron {row['neuron']} is outside layer {row['layer']} (0–{proj.weight.shape[1]-1}).")


def calibrate(backbone, layers, inputs, neurons, cancel):
    """One UNSTEERED pass; measure outgoing residuals, omitting absolute token 0."""
    import torch
    medians, handles = {}, []
    targets = sorted({x["layer"] for x in neurons if x["mode"] == "add" and x["dose"] != 0})
    if not targets:
        return medians
    if inputs["input_ids"].shape[-1] < 2:
        raise ValueError("Residual calibration needs at least two input tokens.")
    def capture(index):
        def hook(_module, _args, output):
            if cancel.is_set():
                raise Cancelled()
            hidden = output[0] if isinstance(output, tuple) else output
            norms = hidden[:, 1:, :].float().norm(dim=-1)
            mask = inputs.get("attention_mask")
            if mask is not None:
                norms = norms[mask[:, 1:].bool()]
            if norms.numel() == 0:
                raise ValueError("No non-padding tokens available for residual calibration.")
            median = float(norms.median().item())
            if not torch.isfinite(torch.tensor(median)) or median <= 0:
                raise ValueError(f"Layer {index} produced an invalid residual norm.")
            medians[index] = median
        return hook
    try:
        for index in targets:
            handles.append(layers[index].register_forward_hook(capture(index)))
        with torch.inference_mode():
            backbone(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return medians


@contextmanager
def steering_hooks(layers, neurons, medians):
    """Modify silu(gate)*up BEFORE down_proj, on every prefill/decode call."""
    import torch
    grouped, metrics, handles = {}, [], []
    for row in neurons:
        layer, neuron, dose = row["layer"], row["neuron"], row["dose"]
        proj = layers[layer].mlp.down_proj
        column_norm = float(proj.weight[:, neuron].float().norm().item())
        if row["mode"] == "add" and dose != 0:
            if not column_norm > 1e-12 or not torch.isfinite(torch.tensor(column_norm)):
                raise ValueError(f"L{layer} N{neuron} has a zero or invalid down-projection column.")
            delta = dose * medians[layer] / column_norm
        else:
            delta = dose
        grouped.setdefault(layer, []).append((neuron, row["mode"], delta))
        metrics.append({**row, "residual_median": medians.get(layer), "column_norm": column_norm,
                        "activation_delta": delta if row["mode"] == "add" else None})
    def hook_for(rows):
        def hook(_module, args):
            h = args[0].clone()
            for neuron, mode, delta in rows:
                if mode == "mul":
                    h[..., neuron] *= delta
                else:
                    h[..., neuron] += delta
            return (h, *args[1:])
        return hook
    try:
        for layer, rows in grouped.items():
            handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook_for(rows)))
        yield metrics
    finally:
        for handle in handles:
            handle.remove()


class Runtime:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = self.tokenizer = self.backbone = None
        self.model_info = None
        self.device = None
        self.state = "idle"
        self.error = None

    def status(self):
        layers = getattr(self.backbone, "layers", [])
        widths = []
        for layer in layers:
            p = getattr(getattr(layer, "mlp", None), "down_proj", None)
            widths.append(p.weight.shape[1] if p is not None and hasattr(p, "weight") and p.weight.ndim == 2 else None)
        return {"state": self.state, "model": self.model_info, "device": self.device,
                "layers": len(layers), "widths": widths,
                "steering_supported": bool(widths) and all(x is not None for x in widths), "error": self.error}

    def unload(self):
        if not self.lock.acquire(blocking=False):
            raise ValueError("The model is busy. Stop the current operation first.")
        try:
            self._clear()
        finally:
            self.lock.release()

    def _clear(self):
        self.model = self.tokenizer = self.backbone = self.model_info = None
        self.device = None
        self.state, self.error = "idle", None
        gc.collect()
        if importlib.util.find_spec("torch"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def load(self, info, device="auto", emit=lambda _: None):
        if device not in ("auto", "mps", "cuda", "cpu"):
            raise ValueError("Choose auto, mps, cuda or cpu.")
        if not dependencies()["ready"]:
            raise ValueError("Inference dependencies are missing or too old. Run `uv sync` in the app folder, then start with `uv run python -m ntweak`.")
        if not self.lock.acquire(blocking=False):
            raise ValueError("The model is busy. Stop the current operation first.")
        try:
            self._clear()
            self.state = "loading"
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            if device == "mps" and not torch.backends.mps.is_available():
                raise ValueError("Apple Metal (MPS) is unavailable on this machine.")
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable on this machine.")
            path = info["path"]
            emit({"status": "loading", "message": f"Reading local weights on {device.upper()}…"})
            config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            if getattr(config, "quantization_config", None):
                raise ValueError("Use unquantized Transformers weights for exact neuron steering. GGUF, MLX, GPTQ and AWQ inference are not supported.")
            cls = AutoModelForCausalLM
            if config.model_type == "qwen3_5":
                from transformers import Qwen3_5ForConditionalGeneration
                cls = Qwen3_5ForConditionalGeneration
            elif config.model_type == "qwen3_5_moe":
                raise ValueError("Mixture-of-experts neuron steering is not supported. Choose Qwen/Qwen3.5-4B.")
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            model = cls.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                        dtype=dtype, attn_implementation="eager").to(device).eval()
            self.model, self.tokenizer, self.backbone = model, tokenizer, find_backbone(model)
            self.model_info, self.device = info, device
            self.state, self.error = "ready", None
            return self.status()
        except Exception as exc:
            self._clear()
            self.state, self.error = "error", str(exc)
            raise
        finally:
            self.lock.release()

    def chat(self, body, cancel, emit):
        if not self.lock.acquire(blocking=False):
            raise ValueError("The model is busy. Wait for the other operation to finish.")
        try:
            if self.model is None:
                raise ValueError("Download and load a model in the Models panel first.")
            import torch
            from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer
            recipe = normalize_recipe(body.get("recipe", {}))
            rows = [x for x in recipe["neurons"] if x["enabled"]] if body.get("steering", True) else []
            if rows and recipe["model"] != self.model_info["repo"]:
                raise ValueError(f"This recipe targets {recipe['model']}. Select that model, or explicitly change the recipe's model ID before applying it.")
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages or len(messages) > 200:
                raise ValueError("Provide between 1 and 200 chat messages.")
            for m in messages:
                if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
                    raise ValueError("Invalid conversation messages.")
            max_context = integer(body.get("max_context", 4096), "Context limit", 256, 32768)
            tok = self.tokenizer
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=bool(body.get("thinking", False)))
            inputs = tok(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(self.device)
            n_input = inputs["input_ids"].shape[-1]
            sampling = recipe["sampling"]
            model_config = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") else self.model.config
            model_context = getattr(model_config, "max_position_embeddings", max_context)
            if n_input + sampling["max_new_tokens"] > min(max_context, model_context):
                raise ValueError(f"Conversation uses {n_input} tokens. With {sampling['max_new_tokens']} output tokens it exceeds the {min(max_context, model_context)} token limit. Start a new chat, reduce output length, or raise the context limit.")
            layers = getattr(self.backbone, "layers", [])
            validate_targets(layers, rows)
            if cancel.is_set():
                raise Cancelled()
            self.state = "generating"
            emit({"type": "phase", "message": "Measuring unsteered residual norms…" if any(x["mode"] == "add" and x["dose"] != 0 for x in rows) else "Generating locally…"})
            medians = calibrate(self.backbone, layers, inputs, rows, cancel) if rows else {}
            if cancel.is_set():
                raise Cancelled()
            class Stop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return cancel.is_set()
            class Stream(TextStreamer):
                def on_finalized_text(self, text, stream_end=False):
                    if text:
                        emit({"type": "token", "text": text})
            streamer = Stream(tok, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(max_new_tokens=sampling["max_new_tokens"], do_sample=sampling["temperature"] > 0,
                          repetition_penalty=sampling["repetition_penalty"], use_cache=True,
                          streamer=streamer, stopping_criteria=StoppingCriteriaList([Stop()]),
                          pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            if sampling["temperature"] > 0:
                kwargs.update(temperature=sampling["temperature"], top_p=sampling["top_p"], min_p=sampling["min_p"], top_k=0)
            # Avoid allocating [prompt length, vocabulary] logits for 248k-token vocabularies.
            if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
                kwargs["logits_to_keep"] = 1
            # Serial model lock makes the RNG seed reproducible without concurrent mutations.
            torch.manual_seed(sampling["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sampling["seed"])
            started = time.monotonic()
            with steering_hooks(layers, rows, medians) as metrics, torch.inference_mode():
                emit({"type": "calibration", "neurons": metrics, "input_tokens": n_input,
                      "calibration": "Current conversation; unsteered; outgoing layer residuals; token 0 excluded"})
                output = self.model.generate(**inputs, **kwargs)
            n_output = output.shape[-1] - n_input
            seconds = time.monotonic() - started
            emit({"type": "done", "input_tokens": n_input, "output_tokens": n_output,
                  "seconds": round(seconds, 2), "tokens_per_second": round(n_output / max(seconds, 0.001), 2),
                  "cancelled": cancel.is_set(), "model": self.model_info["repo"], "recipe": recipe})
        finally:
            if self.model is not None:
                self.state = "ready"
            self.lock.release()
