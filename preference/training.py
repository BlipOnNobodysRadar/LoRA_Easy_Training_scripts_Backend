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
from .methods import method_settings, signature_settings
from .objectives import objective_loss, leco_inputs, set_multiplier
from .optimization import build_optimization


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def micro_seed(seed, step, micro):
    return int(hashlib.sha256(f"{seed}:{step}:{micro}".encode()).hexdigest()[:15], 16)


def eligible_records(store, reference_id, allow_synthetic, objective="dpo"):
    rows = store.list_comparisons(status="eligible")
    rows = [r for r in rows if allow_synthetic or not r.get("synthetic", False)]
    if not rows:
        raise ValueError("No eligible rated pairs. Save an explicit A/B preference; ties and skips are retained but not trained.")
    if len({bool(r.get("synthetic", False)) for r in rows}) != 1:
        raise ValueError("Synthetic tests and real preferences must be in separate datasets")
    for row in rows:
        if objective == "addift" and row.get("pair_kind") != "aligned_edit_v1":
            raise ValueError("ADDifT needs explicitly imported aligned edits. Use a separate dataset; ordinary A/B generations are not aligned.")
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


def cache_pairs(model, store, rows, seed, cancelled, posterior_mean=False):
    cached = {}
    for index, row in enumerate(rows):
        if cancelled():
            raise Cancelled()
        images = {image["id"]: image for image in row["images"]}
        winner = row["feedback"]["preference"]
        ordered = [images[winner], images["b" if winner == "a" else "a"]]
        extra = {"posterior_mean": True} if posterior_mean else {}
        latent_list = [model.encode_image(store.root / image["path"], micro_seed(seed, image["sha256"], 0), **extra)
                       for image in ordered]
        settings = row["generation_settings"]
        text, vector = model.encode_prompt(row["prompt"], settings["width"], settings["height"])
        cached[row["id"]] = {"latents": torch.cat(latent_list), "text": text.cpu(), "vector": vector.cpu()}
        print(f"Cached pair {index + 1}/{len(rows)} ({row['split']})", flush=True)
    return cached


def pair_inputs(cached, seed, model, scheduler, min_timestep=0, max_timestep=1000):
    generator = torch.Generator(device=model.device).manual_seed(seed)
    latents = cached["latents"].to(model.device).float()
    timestep = torch.randint(min_timestep, max_timestep, (1,),
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
    method, options = method_settings(settings)
    limits = {k: options[k] for k in ("min_timestep", "max_timestep")} if method == "addift" else {}
    for row in rows:
        inputs = pair_inputs(cached[row["id"]], micro_seed(settings["seed"], row["id"], "validation"), model, scheduler, **limits)
        loss, metrics = objective_loss(model, inputs, settings, row)
        results.append({"loss": loss.item(), **metrics})
    return {key: sum(r[key] for r in results) / len(results) for key in results[0]} if results else None


def restore_optimization(optimizer, lr_scheduler, saved):
    """Restore native state, including the custom optimizer's offloaded moments."""
    if lr_scheduler is not None and saved.get("lr_scheduler") is None:
        raise ValueError("Checkpoint is missing its learning-rate scheduler state")
    optimizer.load_state_dict(saved["optimizer"])
    if optimizer.__class__.__name__ == "SimplifiedAdEMAMixExM":
        # Optimizer.load_state_dict casts floating states to the parameter's dtype
        # and device. This optimizer deliberately keeps moments in separate storage.
        device = torch.device(optimizer.state_storage_device)
        for state in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    value = state[key].to(device=device, dtype=optimizer.state_storage_dtype)
                    state[key] = value.pin_memory() if device.type == "cpu" else value
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(saved["lr_scheduler"])


def save_checkpoint(run_dir, step, model, optimizer, config, signature, metrics, lr_scheduler=None):
    destination = run_dir / f"checkpoint-{step:06d}"
    if destination.exists():
        return destination
    temporary = run_dir / (".checkpoint-" + uuid4().hex)
    temporary.mkdir()
    settings = config["training"]
    method, _ = method_settings(settings)
    model.save_preference(temporary / "preference_lora.safetensors", settings["rank"], settings["alpha"],
                          {"ss_steps": str(step), "preference_synthetic_test": str(settings["allow_synthetic"]).lower(),
                           "preference_objective": {"dpo": "diffusion-dpo-v1", "addift": "sdxl-aligned-addift-v1", "leco": "sdxl-leco-ddim-v1"}[method]})
    torch.save({"optimizer": optimizer.state_dict(), "step": step,
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary / "training_state.pt")
    atomic_json(temporary / "state.json", {"schema_version": 1, "step": step,
                                         "signature": signature, "metrics": metrics, "saved_at": utc_now()})
    os.replace(temporary, destination)
    atomic_json(run_dir / "latest.json", {"checkpoint": destination.name, "step": step})
    print(f"Checkpoint saved: {destination}", flush=True)
    return destination


def train(config, resume=None, cancelled=lambda: False):
    settings = config["training"]
    method, options = method_settings(settings)
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
    if method == "leco":
        rows = [{"id": digest(options), "split": "train", "prompt": options["target"],
                 "synthetic": settings["allow_synthetic"], "feedback": {"strength": "normal"}}]
    else:
        rows = eligible_records(store, identity["reference_id"], settings["allow_synthetic"], method)
    training_rows = [r for r in rows if r["split"] == "train"]
    validation_rows = [r for r in rows if r["split"] == "validation"]
    fixed_settings = signature_settings(settings, config["generation"])
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
    optimizer, lr_scheduler, optimization = build_optimization(parameters, settings)
    atomic_json(run_dir / "optimization.json", optimization)
    if resume:
        saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        if saved["step"] != start_step:
            raise ValueError("Checkpoint step mismatch")
        restore_optimization(optimizer, lr_scheduler, saved)
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    cached = {} if method == "leco" else cache_pairs(model, store, rows, settings["seed"], cancelled, method == "addift")
    scheduler = noise_scheduler()
    limits = {k: options[k] for k in ("min_timestep", "max_timestep")} if method == "addift" else {}

    def inputs_for(row, seed):
        if method == "leco":
            return leco_inputs(model, settings, config["generation"], seed, cancelled)
        return pair_inputs(cached[row["id"]], seed, model, scheduler, **limits)

    # Frozen reference and initial policy must agree for a newly initialized delta.
    probe_inputs = inputs_for(training_rows[0], 1729)
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
                try:
                    inputs = inputs_for(row, seed)
                except Cancelled:
                    status = "stopped"
                    break
                loss, pair_metrics = objective_loss(model, inputs, settings, row, step)
                (loss / settings["gradient_accumulation"]).backward()
                set_multiplier(model, 1.0)
                losses.append(loss.item())
                per_micro.append(pair_metrics)
                del inputs, loss
            if status == "stopped":
                optimizer.zero_grad(set_to_none=True)
                set_multiplier(model, 1.0)
                break
            model.assert_gradient_ownership()
            # A zero limit disables clipping but still checks the gradient norm.
            limit = optimization["max_grad_norm"] or math.inf
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, limit, error_if_nonfinite=True).item()
            if not math.isfinite(grad_norm) or (step == start_step and grad_norm == 0):
                raise FloatingPointError("Preference gradients are zero or non-finite")
            used_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            final_step = step + 1
            metrics = {"step": final_step, "loss": sum(losses) / len(losses), "gradient_norm": grad_norm,
                       "learning_rate": used_lr, "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                       **{key: sum(r[key] for r in per_micro) / len(per_micro) for key in per_micro[0]},
                       "elapsed_seconds": time.monotonic() - started,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            if hasattr(optimizer, "linear_hl_warmup_scheduler"):
                group = optimizer.param_groups[0]
                metrics["momentum_beta1"] = (optimizer.linear_hl_warmup_scheduler(
                    final_step, group["betas"][0], group["min_beta1"], group["beta1_warmup"])
                    if group["beta1_warmup"] else group["betas"][0])
            # Same noising inputs, adapter disabled: updates must not move reference.
            if final_step == start_step + 1 or final_step == settings["max_steps"]:
                with model.reference_mode():
                    check = model.predict(*probe_inputs[:4]).detach()
                metrics["reference_max_diff"] = (check.float() - reference_probe.float()).abs().max().item()
                if metrics["reference_max_diff"] != 0:
                    raise AssertionError("Frozen reference changed during preference training")
            if final_step % settings["checkpoint_every"] == 0 or final_step == settings["max_steps"]:
                metrics["validation"] = evaluate(model, validation_rows, cached, settings, scheduler)
                checkpoint_path = save_checkpoint(run_dir, final_step, model, optimizer, config, signature, metrics, lr_scheduler)
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
            checkpoint_path = save_checkpoint(run_dir, final_step, model, optimizer, config, signature, metrics, lr_scheduler)
        changed = any(not torch.equal(initial_parameters[name], p.detach().cpu())
                      for name, p in model.preference.named_parameters())
        if final_step > start_step and not changed:
            raise AssertionError("No preference parameter changed")
        report = {"status": status, "run_dir": str(run_dir), "step": final_step,
                  "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                  "parameters_changed": changed, "frozen_gradient_check": True,
                  "objective": method, "train_pairs": len(training_rows) if method != "leco" else 0,
                  "validation_pairs": len(validation_rows), "concept_prompts": 1 if method == "leco" else 0,
                  "synthetic": bool(training_rows[0].get("synthetic")),
                  "adapter_reports": model.adapter_reports, **metrics}
        if status == "completed" and checkpoint_path and config["model"]["base_loras"]:
            from .combined import export_combined
            try:
                report.update(export_combined(config, checkpoint_path / "preference_lora.safetensors",
                              run_dir / f"combined_{method.upper()}_step{final_step:06d}.safetensors", cancelled))
            except (ValueError, OSError) as error:
                # An unsupported export must not invalidate a saved/resumable run.
                report["combined_export_error"] = str(error)
                print(f"Training checkpoint saved; combined export unavailable: {error}", flush=True)
        atomic_json(run_dir / "status.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report
