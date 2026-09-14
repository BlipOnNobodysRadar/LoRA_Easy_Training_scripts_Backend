"""Resumable single-GPU LoRA preference training with immutable data snapshots."""

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image
from safetensors.torch import load_file

from .config import atomic_json, file_identity, utc_now, prompt_group
from .loss import diffusion_dpo_loss
from .sdxl import SDXLModel, noise_scheduler, Cancelled
from .store import PreferenceStore


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def micro_seed(seed, step, micro):
    return int(hashlib.sha256(f"{seed}:{step}:{micro}".encode()).hexdigest()[:15], 16)


def eligible_records(store, reference_id, allow_synthetic):
    rows = store.list_comparisons(status="eligible")
    rows = [r for r in rows if allow_synthetic or not r.get("synthetic", False)]
    if not rows:
        raise ValueError("No eligible rated pairs. Save an explicit A/B preference; ties and skips are retained but not trained.")
    if len({bool(r.get("synthetic", False)) for r in rows}) != 1:
        raise ValueError("Synthetic tests and real preferences must be in separate datasets")
    for row in rows:
        if row["group_id"] != prompt_group(row["prompt"]):
            raise ValueError("Prompt group identity is invalid; refusing possible validation leakage")
        if row["model"].get("reference_id") != reference_id:
            raise ValueError("Dataset includes a different reference model. Use a separate dataset for each base/adapter combination.")
        if len(row["images"]) != 2 or {i["id"] for i in row["images"]} != {"a", "b"}:
            raise ValueError("This trainer currently requires exactly two images named a and b")
        sizes = []
        for image in row["images"]:
            path = (store.root / image["path"]).resolve(strict=True)
            if not path.is_relative_to(store.root) or file_identity(path)["sha256"] != image["sha256"]:
                raise ValueError("A comparison image changed or is outside its dataset")
            with Image.open(path) as img:
                sizes.append(img.size)
        expected = (row["generation_settings"]["width"], row["generation_settings"]["height"])
        if sizes != [expected, expected] or any(s % 64 or not 256 <= s <= 1536 for s in expected):
            raise ValueError("Pair image sizes must match recorded dimensions (multiples of 64, 256–1536)")
    train_groups = {r["group_id"] for r in rows if r["split"] == "train"}
    validation_groups = {r["group_id"] for r in rows if r["split"] == "validation"}
    if train_groups & validation_groups:
        raise ValueError("Prompt groups overlap between training and validation")
    if not any(r["split"] == "train" for r in rows):
        raise ValueError("No rated training pairs; the rated pairs are all held out for validation")
    return rows


def cache_pairs(model, store, rows, seed, cancelled):
    cached = {}
    for index, row in enumerate(rows):
        if cancelled():
            raise Cancelled()
        images = {image["id"]: image for image in row["images"]}
        winner = row["feedback"]["preference"]
        ordered = [images[winner], images["b" if winner == "a" else "a"]]
        latent_list = [model.encode_image(store.root / image["path"], micro_seed(seed, image["sha256"], 0))
                       for image in ordered]
        settings = row["generation_settings"]
        text, vector = model.encode_prompt(row["prompt"], settings["width"], settings["height"])
        cached[row["id"]] = {"latents": torch.cat(latent_list), "text": text.cpu(), "vector": vector.cpu()}
        print(f"Cached pair {index + 1}/{len(rows)} ({row['split']})", flush=True)
    return cached


def pair_inputs(cached, seed, model, scheduler):
    generator = torch.Generator(device=model.device).manual_seed(seed)
    latents = cached["latents"].to(model.device).float()
    timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,),
                             generator=generator, device=model.device).repeat(2)
    # Different generation seeds are appropriate for collection. During training,
    # paired images must share this newly sampled diffusion noise and timestep.
    noise = torch.randn((1, *latents.shape[1:]), generator=generator, device=model.device).repeat(2, 1, 1, 1)
    noisy = scheduler.add_noise(latents, noise, timestep)
    text = cached["text"].to(model.device).repeat(2, 1, 1)
    vector = cached["vector"].to(model.device).repeat(2, 1)
    return noisy, timestep, text, vector, noise


def denoising_errors(model, inputs):
    noisy, timestep, text, vector, noise = inputs
    prediction = model.predict(noisy, timestep, text, vector)
    return (prediction.float() - noise).square().mean(dim=(1, 2, 3)).unsqueeze(0)


@torch.no_grad()
def evaluate(model, rows, cached, settings, scheduler):
    results = []
    for row in rows:
        inputs = pair_inputs(cached[row["id"]], micro_seed(settings["seed"], row["id"], "validation"), model, scheduler)
        with model.reference_mode():
            reference = denoising_errors(model, inputs)
        policy = denoising_errors(model, inputs)
        loss, metrics = diffusion_dpo_loss(policy, reference, settings["beta"],
                                          [settings["strength_weights"][row["feedback"]["strength"]]])
        results.append({"loss": loss.item(), **metrics})
    return {key: sum(r[key] for r in results) / len(results) for key in results[0]} if results else None


def save_checkpoint(run_dir, step, model, optimizer, config, signature, metrics):
    destination = run_dir / f"checkpoint-{step:06d}"
    if destination.exists():
        return destination
    temporary = run_dir / (".checkpoint-" + uuid4().hex)
    temporary.mkdir()
    settings = config["training"]
    model.save_preference(temporary / "preference_lora.safetensors", settings["rank"], settings["alpha"],
                          {"ss_steps": str(step), "preference_synthetic_test": str(settings["allow_synthetic"]).lower()})
    torch.save({"optimizer": optimizer.state_dict(), "step": step,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary / "training_state.pt")
    atomic_json(temporary / "state.json", {"schema_version": 1, "step": step,
                                         "signature": signature, "metrics": metrics, "saved_at": utc_now()})
    os.replace(temporary, destination)
    atomic_json(run_dir / "latest.json", {"checkpoint": destination.name, "step": step})
    print(f"Checkpoint saved: {destination}", flush=True)
    return destination


def train(config, resume=None, cancelled=lambda: False):
    settings = config["training"]
    if config["model"].get("preference_lora") and config["model"]["preference_weight"] != 1.0:
        raise ValueError("Training an existing preference adapter requires preference_weight=1; other weights are inference-only")
    if settings["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # cuDNN SDPA backward is not deterministic on the supported torch stack.
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.use_deterministic_algorithms(settings["deterministic"])
    torch.backends.cudnn.deterministic = settings["deterministic"]
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    store = PreferenceStore(config["dataset_dir"])
    # Hash/model/data validation precedes GPU allocation.
    from .config import model_identity
    identity = model_identity(config["model"])
    rows = eligible_records(store, identity["reference_id"], settings["allow_synthetic"])
    training_rows = [r for r in rows if r["split"] == "train"]
    validation_rows = [r for r in rows if r["split"] == "validation"]
    fixed_settings = {k: v for k, v in settings.items() if k not in ("max_steps", "checkpoint_every", "output_dir")}
    signature = digest({"reference_id": identity["reference_id"], "settings": fixed_settings, "data": rows})
    if resume:
        checkpoint = Path(resume).resolve(strict=True)
        state = json.loads((checkpoint / "state.json").read_text())
        if state["signature"] != signature:
            raise ValueError("Resume configuration or rated dataset changed. Restore them or start a new run using the saved adapter.")
        run_dir = checkpoint.parent
        start_step = state["step"]
        adapter_path = checkpoint / "preference_lora.safetensors"
        if any(int(p.name.split("-")[1]) > start_step for p in run_dir.glob("checkpoint-[0-9]*") if p.is_dir()):
            raise ValueError("A later checkpoint already exists. Resume the latest checkpoint to avoid a divergent overwrite.")
    else:
        run_dir = Path(settings["output_dir"]) / ("run-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir / "config.json", config)
        atomic_json(run_dir / "dataset_snapshot.json", rows)
        start_step = 0
        adapter_path = config["model"].get("preference_lora")
    if start_step >= settings["max_steps"]:
        raise ValueError("max_steps must exceed the saved step when resuming")
    if cancelled():
        raise Cancelled()
    model = SDXLModel(config["model"])
    torch.cuda.reset_peak_memory_stats()
    model.attach_preference(settings["rank"], settings["alpha"], path=adapter_path,
                            trainable=True, weight=1.0)
    parameters = list(model.preference.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=0.0)
    if resume:
        saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        if saved["step"] != start_step:
            raise ValueError("Checkpoint step mismatch")
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    cached = cache_pairs(model, store, rows, settings["seed"], cancelled)
    scheduler = noise_scheduler()
    # Frozen reference and initial policy must agree for a newly initialized delta.
    probe_inputs = pair_inputs(cached[training_rows[0]["id"]], 1729, model, scheduler)
    with model.reference_mode():
        reference_probe = model.predict(*probe_inputs[:4]).detach().clone()
    with torch.no_grad():
        policy_probe = model.predict(*probe_inputs[:4]).detach()
    zero_difference = (reference_probe.float() - policy_probe.float()).abs().max().item()
    if not adapter_path and zero_difference != 0:
        raise AssertionError(f"Zero preference adapter changed the starting model: {zero_difference}")
    started = time.monotonic()
    metrics = {"initial_policy_reference_max_diff": zero_difference}
    final_step = start_step
    status = "completed"
    checkpoint_path = None
    initial_parameters = {name: p.detach().cpu().clone() for name, p in model.preference.named_parameters()}
    try:
        for step in range(start_step, settings["max_steps"]):
            if cancelled():
                status = "stopped"
                break
            optimizer.zero_grad(set_to_none=True)
            losses = []
            per_micro = []
            for micro in range(settings["gradient_accumulation"]):
                seed = micro_seed(settings["seed"], step, micro)
                row = training_rows[random.Random(seed).randrange(len(training_rows))]
                inputs = pair_inputs(cached[row["id"]], seed, model, scheduler)
                with model.reference_mode():
                    reference_errors = denoising_errors(model, inputs)
                inputs[0].requires_grad_(True)  # native reentrant gradient checkpointing needs a grad input
                policy_errors = denoising_errors(model, inputs)
                loss, pair_metrics = diffusion_dpo_loss(policy_errors, reference_errors, settings["beta"],
                                                       [settings["strength_weights"][row["feedback"]["strength"]]])
                (loss / settings["gradient_accumulation"]).backward()
                losses.append(loss.item())
                per_micro.append(pair_metrics)
                del inputs, policy_errors, reference_errors, loss
            model.assert_gradient_ownership()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True).item()
            if not math.isfinite(grad_norm) or (step == start_step and grad_norm == 0):
                raise FloatingPointError("Preference gradients are zero or non-finite")
            optimizer.step()
            final_step = step + 1
            metrics = {"step": final_step, "loss": sum(losses) / len(losses), "gradient_norm": grad_norm,
                       **{key: sum(r[key] for r in per_micro) / len(per_micro) for key in per_micro[0]},
                       "elapsed_seconds": time.monotonic() - started,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            # Same noising inputs, adapter disabled: updates must not move reference.
            if final_step == start_step + 1 or final_step == settings["max_steps"]:
                with model.reference_mode():
                    check = model.predict(*probe_inputs[:4]).detach()
                metrics["reference_max_diff"] = (check.float() - reference_probe.float()).abs().max().item()
                if metrics["reference_max_diff"] != 0:
                    raise AssertionError("Frozen reference changed during preference training")
            if final_step % settings["checkpoint_every"] == 0 or final_step == settings["max_steps"]:
                metrics["validation"] = evaluate(model, validation_rows, cached, settings, scheduler)
                checkpoint_path = save_checkpoint(run_dir, final_step, model, optimizer, config, signature, metrics)
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            atomic_json(run_dir / "status.json", {"status": "running", **metrics})
            print(json.dumps(metrics), flush=True)
    except BaseException as error:
        status = "failed"
        atomic_json(run_dir / "status.json", {"status": status, "step": final_step, "error": str(error)})
        raise
    else:
        if final_step > start_step:
            checkpoint_path = save_checkpoint(run_dir, final_step, model, optimizer, config, signature, metrics)
        changed = any(not torch.equal(initial_parameters[name], p.detach().cpu())
                      for name, p in model.preference.named_parameters())
        if final_step > start_step and not changed:
            raise AssertionError("No preference parameter changed")
        report = {"status": status, "run_dir": str(run_dir), "step": final_step,
                  "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                  "parameters_changed": changed, "frozen_gradient_check": True,
                  "train_pairs": len(training_rows), "validation_pairs": len(validation_rows),
                  "synthetic": bool(training_rows[0].get("synthetic")),
                  "adapter_reports": model.adapter_reports, **metrics}
        atomic_json(run_dir / "status.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report
