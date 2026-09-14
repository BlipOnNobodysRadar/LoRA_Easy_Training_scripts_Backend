"""SDXL edit objectives using the same frozen checkpoint and base adapter stack.

ADDifT-style: match the frozen rejected-image prediction at the preferred-image
latent. Reversing both images and the adapter sign teaches the inverse direction.
LECO: use sd-scripts' target formula with partial DDIM reference trajectories.
These are explicit variants, not numerical reproductions of external trainers.
"""

import torch

from .loss import diffusion_dpo_loss
from .methods import method_settings


def set_multiplier(model, value):
    # Leave this value active through backward: gradient checkpointing recomputes
    # the forward, including LoRA's global multiplier.
    model.preference_weight = value
    model.preference.set_multiplier(value)


def prediction_error(model, inputs):
    return (model.predict(*inputs[:4]).float() - inputs[4]).square().mean((1, 2, 3)).unsqueeze(0)


def objective_loss(model, inputs, settings, row, step=0):
    method, options = method_settings(settings)
    weight = settings["strength_weights"][row["feedback"]["strength"]]
    if method == "dpo":
        with model.reference_mode():
            reference = prediction_error(model, inputs)
        inputs[0].requires_grad_(True)
        policy = prediction_error(model, inputs)
        return diffusion_dpo_loss(policy, reference, settings["beta"], [weight])
    if method == "addift":
        reverse = options["alternate_inverse"] and step % 2 == 1
        student, teacher = (1, 0) if reverse else (0, 1)  # inputs are winner first
        set_multiplier(model, -1.0 if reverse else 1.0)
        with model.reference_mode():
            target = model.predict(*(x[teacher:teacher+1] for x in inputs[:4])).float()
        policy_inputs = [x[student:student+1] for x in inputs[:4]]
        policy_inputs[0] = policy_inputs[0].detach().requires_grad_(True)
        predicted = model.predict(*policy_inputs).float()
        error = (predicted - target).square().mean()
        return error * weight, {"edit_mse": error.item(), "inverse_direction": float(reverse)}
    from library.leco_train_util import PromptSettings
    with model.reference_mode():
        predictions = [model.predict(inputs[0], inputs[1], *inputs[4][name]).float()
                       for name in ("positive", "neutral", "unconditional")]
        formula = PromptSettings(target=options["target"], action=options["action"],
                                 guidance_scale=options["guidance_scale"])
        target = formula.build_target(*predictions)
    inputs[0].requires_grad_(True)
    predicted = model.predict(*inputs[:4]).float()
    error = (predicted - target).square().mean()
    return error, {"concept_mse": error.item()}


@torch.no_grad()
def leco_inputs(model, training, generation, seed, cancelled=lambda: False):
    from .sdxl import DDIMScheduler, noise_scheduler, Cancelled
    _, settings = method_settings(training)
    width, height = generation["width"], generation["height"]
    embeddings = {name: model.encode_prompt(settings[name], width, height)
                  for name in ("target", "positive", "neutral", "unconditional")}
    scheduler = DDIMScheduler.from_config(noise_scheduler().config)
    scheduler.set_timesteps(settings["denoising_steps"], device=model.device)
    rng = torch.Generator(device=model.device).manual_seed(seed)
    prefix = torch.randint(1, settings["denoising_steps"], (1,), generator=rng, device=model.device).item()
    latents = torch.randn((1, 4, height // 8, width // 8), generator=rng, device=model.device,
                          dtype=model.dtype) * scheduler.init_noise_sigma
    text = torch.cat([embeddings["unconditional"][0], embeddings["target"][0]])
    vector = torch.cat([embeddings["unconditional"][1], embeddings["target"][1]])
    with model.reference_mode():
        for timestep in scheduler.timesteps[:prefix]:
            if cancelled():
                raise Cancelled()
            noisy = scheduler.scale_model_input(latents.repeat(2, 1, 1, 1), timestep)
            uncond, cond = model.predict(noisy, timestep.expand(2), text, vector).chunk(2)
            prediction = uncond + settings["denoise_cfg"] * (cond - uncond)
            latents = scheduler.step(prediction, timestep, latents, eta=0, return_dict=False)[0]
    return (latents.detach(), scheduler.timesteps[prefix].reshape(1), *embeddings["target"], embeddings)
