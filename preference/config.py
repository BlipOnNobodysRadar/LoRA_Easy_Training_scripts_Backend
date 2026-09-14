"""Torch-free configuration, provenance, and filesystem helpers."""

import copy
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULTS = {
    "schema_version": 1,
    "dataset_dir": "preference_data",
    "model": {"family": "sdxl", "checkpoint": "", "base_loras": [],
              "preference_lora": None, "preference_weight": 1.0, "max_token_length": 225},
    "generation": {"prompts": [], "negative_prompt": "", "width": 1024, "height": 1024,
                   "steps": 25, "cfg": 7.0, "sampler": "euler", "seed": 9796,
                   "pairs_per_prompt": 1, "validation_fraction": 0.2, "split_seed": 9796},
    "training": {"output_dir": "preference_runs", "rank": 16, "alpha": 16,
                 "learning_rate": 1e-5, "beta": 5000.0, "max_steps": 50,
                 "gradient_accumulation": 4, "checkpoint_every": 25, "seed": 9796,
                 "strength_weights": {"slight": 0.5, "normal": 1.0, "strong": 2.0},
                 "allow_synthetic": False, "deterministic": True},
}
_HASH_CACHE = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        _HASH_CACHE[key] = digest.hexdigest()
    return {"path": str(path), "sha256": _HASH_CACHE[key], "bytes": stat.st_size}


def model_identity(model):
    result = {"family": "sdxl", "checkpoint": file_identity(model["checkpoint"]),
              "base_loras": [{**file_identity(item["path"]), "weight": item["weight"]}
                             for item in model["base_loras"]],
              "max_token_length": model["max_token_length"],
              "conditioning": "sd-scripts-weighted-sdxl-v1", "prediction_type": "epsilon",
              "vae_encoding": "checkpoint-fp32-sampled-v1"}
    # Paths are descriptive; the reference is identified by content and order/scales.
    signature = {"checkpoint": result["checkpoint"]["sha256"],
                 "base_loras": [(x["sha256"], x["weight"]) for x in result["base_loras"]],
                 "conditioning": result["conditioning"], "max_token_length": result["max_token_length"],
                 "prediction_type": result["prediction_type"], "vae_encoding": result["vae_encoding"]}
    result["reference_id"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if model.get("preference_lora"):
        result["preference_lora"] = {**file_identity(model["preference_lora"]),
                                     "weight": model["preference_weight"]}
    else:
        result["preference_lora"] = None
    return result


def load_config(path):
    path = Path(path).resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or raw.get("schema_version", 1) != 1:
        raise ValueError("Expected a version 1 preference configuration object")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(raw)
    for key in ("model", "generation", "training"):
        cfg[key] = {**copy.deepcopy(DEFAULTS[key]), **raw.get(key, {})}
    cfg["training"]["strength_weights"] = {
        **DEFAULTS["training"]["strength_weights"], **cfg["training"]["strength_weights"]}
    if cfg["model"]["family"] != "sdxl":
        raise ValueError("Only standard epsilon-prediction SDXL checkpoints are supported in this version")

    def resolved(value, existing=False):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("A required path is empty")
        p = Path(value).expanduser()
        p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
        if existing and (not p.is_file() or p.suffix.lower() != ".safetensors"):
            raise ValueError(f"Expected an existing .safetensors file: {p}")
        return str(p)

    cfg["dataset_dir"] = resolved(cfg["dataset_dir"])
    cfg["training"]["output_dir"] = resolved(cfg["training"]["output_dir"])
    cfg["model"]["checkpoint"] = resolved(cfg["model"]["checkpoint"], True)
    for item in cfg["model"]["base_loras"]:
        item["path"] = resolved(item["path"], True)
        item["weight"] = finite_number(item.get("weight", 1.0), "base adapter weight")
    if cfg["model"].get("preference_lora"):
        cfg["model"]["preference_lora"] = resolved(cfg["model"]["preference_lora"], True)
    cfg["model"]["preference_weight"] = finite_number(cfg["model"]["preference_weight"], "preference weight")
    for section, names in (("generation", ("width", "height", "steps", "pairs_per_prompt")),
                           ("training", ("rank", "max_steps", "gradient_accumulation", "checkpoint_every"))):
        for name in names:
            value = cfg[section][name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{section}.{name} must be a positive integer")
    g, t = cfg["generation"], cfg["training"]
    if any(g[key] % 64 or not 256 <= g[key] <= 1536 for key in ("width", "height")):
        raise ValueError("Image dimensions must be multiples of 64 between 256 and 1536")
    if g["sampler"] not in ("euler", "ddim", "dpmpp_2m"):
        raise ValueError("Sampler must be euler, ddim, or dpmpp_2m")
    if not isinstance(g["prompts"], list) or any(not isinstance(p, str) or not p.strip() for p in g["prompts"]):
        raise ValueError("prompts must be a list of nonempty strings")
    if not isinstance(g["negative_prompt"], str):
        raise ValueError("negative_prompt must be a string")
    for name in ("seed", "split_seed"):
        if type(g[name]) is not int or not 0 <= g[name] < 2**63:
            raise ValueError(f"generation.{name} must be an integer from 0 through 2^63-1")
    if type(t["seed"]) is not int or not 0 <= t["seed"] < 2**63:
        raise ValueError("training.seed must be an integer from 0 through 2^63-1")
    for name in ("alpha", "learning_rate", "beta"):
        t[name] = finite_number(t[name], name, positive=True)
    for key, value in t["strength_weights"].items():
        t["strength_weights"][key] = finite_number(value, "strength weight", positive=True)
    g["cfg"] = finite_number(g["cfg"], "CFG", positive=True)
    g["validation_fraction"] = finite_number(g["validation_fraction"], "validation_fraction")
    if not 0 <= g["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 (inclusive) and 1")
    if cfg["model"]["max_token_length"] not in (75, 150, 225):
        raise ValueError("max_token_length must be 75, 150, or 225")
    if type(t["allow_synthetic"]) is not bool:
        raise ValueError("allow_synthetic must be true or false")
    if type(t["deterministic"]) is not bool:
        raise ValueError("deterministic must be true or false")
    return cfg


def finite_number(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def prompt_group(prompt):
    return hashlib.sha256(" ".join(prompt.casefold().split()).encode()).hexdigest()


def split_for_prompt(prompt, fraction, seed):
    group = prompt_group(prompt)
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    return group, "validation" if value < fraction else "train"
