"""Generate immutable, reproducible comparisons for later human review."""

import hashlib
import json
import platform
import time
from uuid import uuid4

import diffusers
import torch
from safetensors import safe_open
from PIL.PngImagePlugin import PngInfo

from .config import atomic_json, file_identity, model_identity, prompt_entries, split_for_prompt, utc_now
from .sdxl import SDXLModel, Cancelled
from .store import PreferenceStore


def generate(config, cancelled=lambda: False):
    settings = config["generation"]
    entries = prompt_entries(settings)
    if not entries:
        raise ValueError("Add at least one generation prompt")
    store = PreferenceStore(config["dataset_dir"])
    existing_groups = {}
    existing = store.list_comparisons()
    batch_seed = (settings["seed"] + len(existing) * 2) % (2**63)
    for row in existing:
        if row["group_id"] in existing_groups and existing_groups[row["group_id"]] != row["split"]:
            raise ValueError("Dataset already contains conflicting prompt splits")
        existing_groups[row["group_id"]] = row["split"]
    identity = model_identity(config["model"])
    if any(row["model"].get("reference_id") != identity["reference_id"] for row in existing):
        raise ValueError("This dataset belongs to another reference. Select a separate dataset directory.")
    if any(bool(row.get("synthetic")) != config["training"]["allow_synthetic"] for row in existing):
        raise ValueError("Synthetic test collections and human collections require separate dataset directories.")
    model = SDXLModel(config["model"])
    if config["model"].get("preference_lora"):
        with safe_open(config["model"]["preference_lora"], framework="pt", device="cpu") as file:
            if (file.metadata() or {}).get("preference_synthetic_test") == "true" and not config["training"]["allow_synthetic"]:
                raise ValueError("This adapter was trained on synthetic test labels. Use a separate test dataset and allow_synthetic=true.")
        model.attach_preference(path=config["model"]["preference_lora"],
                                weight=config["model"]["preference_weight"])
    session = uuid4().hex
    session_path = store.root / "sessions" / (session + ".json")
    manifest = {"id": session, "created_at": utc_now(), "config": config,
                "model": model.identity, "pairs": [], "status": "running"}
    atomic_json(session_path, manifest)
    total = sum(entry["pairs"] for entry in entries)
    pair_number = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        for prompt_index, entry in enumerate(entries):
            prompt, negative = entry["prompt"], entry["negative_prompt"]
            pair_settings = {key: value for key, value in settings.items() if key != "prompts"}
            pair_settings.update(negative_prompt=negative, pairs_per_prompt=entry["pairs"])
            group, split = split_for_prompt(prompt, settings["validation_fraction"], settings["split_seed"])
            split = existing_groups.get(group, split)
            for pair_index in range(entry["pairs"]):
                if cancelled():
                    raise Cancelled()
                pair_id = uuid4().hex
                print(f"Comparison {pair_number + 1}/{total}: {prompt}", flush=True)
                record = {"id": pair_id, "created_at": utc_now(), "session_id": session,
                          "group_id": group, "split": split, "synthetic": config["training"]["allow_synthetic"],
                          "prompt": prompt, "negative_prompt": negative,
                          "model": model.identity, "images": [],
                          "generation_settings": {**pair_settings, "engine": "native-sdxl-sdpa-v1",
                              "prompt_index": prompt_index, "pair_index": pair_index,
                              "batch_seed": batch_seed,
                              "torch": torch.__version__, "diffusers": diffusers.__version__,
                              "python": platform.python_version(), "gpu": torch.cuda.get_device_name(),
                              "precision": "bf16-unet-fp32-vae", "vae_scale_factor": 0.13025}}
                # A/B side assignment is randomized reproducibly; inference settings
                # are identical. The image IDs, actual seeds, and order are retained.
                seeds = [(batch_seed + pair_number * 2 + side) % (2**63) for side in range(2)]
                if int(hashlib.sha256(f"{batch_seed}:{pair_number}:order".encode()).hexdigest()[:2], 16) % 2:
                    seeds.reverse()
                for side, seed in zip(("a", "b"), seeds):
                    image, scheduler_config = model.generate(prompt, negative, pair_settings, seed, cancelled)
                    record["generation_settings"]["scheduler_config"] = scheduler_config
                    relative = f"images/{pair_id}-{side}.png"
                    image_path = store.root / relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    metadata = PngInfo()
                    metadata.add_text("preference_generation", json.dumps({
                        "prompt": prompt, "negative_prompt": negative, "seed": seed,
                        "model": model.identity, "settings": record["generation_settings"]}))
                    with image_path.open("xb") as handle:
                        image.save(handle, format="PNG", pnginfo=metadata)
                    record["images"].append({"id": side, "path": relative, "seed": seed,
                                              "sha256": file_identity(image_path)["sha256"]})
                store.add_comparison(record)
                manifest["pairs"].append(pair_id)
                atomic_json(session_path, manifest)
                pair_number += 1
        manifest["status"] = "completed"
    except Cancelled:
        manifest["status"] = "stopped"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic() - started,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        atomic_json(session_path, manifest)
    report = {"status": manifest["status"], "dataset_dir": str(store.root),
              "session": str(session_path), "comparisons": len(manifest["pairs"]),
              "peak_allocated_gib": manifest["peak_allocated_gib"],
              "peak_reserved_gib": manifest["peak_reserved_gib"]}
    print(json.dumps(report, indent=2), flush=True)
    return report
