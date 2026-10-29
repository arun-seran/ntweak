"""Recipe interchange and strict validation. No model dependencies required."""
from __future__ import annotations

import json
import math
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen

NEURONARY_API = "https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api"


def number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a number.") from None
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return result


def integer(value, name, low, high):
    result = number(value, name, low, high)
    if result != int(result):
        raise ValueError(f"{name} must be a whole number.")
    return int(result)


def normalize_recipe(data):
    if isinstance(data, list):
        data = {"neurons": data}
    if not isinstance(data, dict):
        raise ValueError("A recipe must be a JSON object or a list of neurons.")
    if isinstance(data.get("recipe"), dict):
        data = data["recipe"]
    rows = data.get("neurons", data.get("steer_neurons", data.get("tweaks", [])))
    if not isinstance(rows, list) or len(rows) > 128:
        raise ValueError("A recipe must contain at most 128 neurons.")
    method = data.get("method") or {}
    if not isinstance(method, dict):
        method = {}
    mode = data.get("mode", method.get("op", "add"))
    neurons = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each neuron must be an object with layer, neuron and dose.")
        op = row.get("mode", mode)
        if op not in ("add", "mul"):
            raise ValueError(f"Unsupported operation {op!r}; use add or mul.")
        neurons.append({
            "layer": integer(row.get("layer"), "Layer", 0, 1000),
            "neuron": integer(row.get("neuron", row.get("n")), "Neuron", 0, 1000000),
            "dose": number(row.get("dose", 0.8), "Dose", -100, 100),
            "mode": op,
            "enabled": row.get("enabled", True) is not False,
            "label": str(row.get("label", ""))[:160],
        })
    sampling = data.get("sampling") or {}
    if not isinstance(sampling, dict):
        raise ValueError("Sampling must be an object.")
    sampling = {
        "temperature": number(sampling.get("temperature", 0.7), "Temperature", 0, 2),
        "top_p": number(sampling.get("top_p", 0.95), "Top p", 0.01, 1),
        "min_p": number(sampling.get("min_p", 0.05), "Min p", 0, 1),
        "repetition_penalty": number(sampling.get("repetition_penalty", 1.1), "Repetition penalty", 0.1, 3),
        "max_new_tokens": integer(sampling.get("max_new_tokens", sampling.get("max_new", 512)), "Max output tokens", 1, 4096),
        "seed": integer(sampling.get("seed", 42), "Seed", 0, 2**32 - 1),
    }
    return {"name": str(data.get("name", "Untitled recipe"))[:200],
            "description": str(data.get("description", ""))[:10000],
            "model": str(data.get("model", "Qwen/Qwen3.5-4B"))[:250],
            "source": str(data.get("source", ""))[:1000],
            "neurons": neurons, "sampling": sampling}


def recipe_id(value):
    value = str(value).strip()
    if "://" in value:
        url = urlparse(value)
        if url.scheme != "https" or url.hostname not in ("neuronary.org", "www.neuronary.org"):
            raise ValueError("Use a neuronary.org recipe link or a recipe ID.")
        match = re.fullmatch(r"/r/([A-Za-z0-9_-]{1,80})/?", url.path)
        if not match:
            raise ValueError("Expected a link like https://neuronary.org/r/9j43ped5.")
        value = match[1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("Invalid recipe ID.")
    return value


def fetch_recipe(value):
    rid = recipe_id(value)
    request = Request(f"{NEURONARY_API}/recipes/{rid}", headers={"Accept": "application/json", "User-Agent": "Ntweak/0.1"})
    with urlopen(request, timeout=30) as response:
        data = json.load(response)
    recipe = normalize_recipe(data)
    recipe["source"] = f"https://neuronary.org/r/{rid}"
    return recipe
