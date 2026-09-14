"""SDXL implementation using this checkout's native checkpoint/adapter loaders.

Base adapters stay frozen and active. Only the separate preference adapter is
switched off for reference predictions. Neither checkpoints nor input adapters
are merged or rewritten on disk.
"""

import gc
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SD_SCRIPTS = Path(__file__).resolve().parents[1] / "sd_scripts"
if str(SD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SD_SCRIPTS))

from diffusers import DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from library import sdxl_model_util, sdxl_train_util, strategy_sdxl, train_util
from networks import lora

from .config import model_identity


class Cancelled(Exception):
    """Cooperative user cancellation, checked between bounded GPU operations."""


def noise_scheduler():
    return DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                         beta_schedule="scaled_linear", clip_sample=False, prediction_type="epsilon")


class SDXLModel:
    def __init__(self, config):
        if not torch.cuda.is_available():
            raise RuntimeError("SDXL preference generation/training requires CUDA")
        self.config = config
        self.identity = model_identity(config)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.base_adapters = []
        self.preference = None
        self.text_cache = {}
        self.adapter_reports = []
        checkpoint = config["checkpoint"]
        print(f"Loading SDXL checkpoint: {checkpoint}", flush=True)
        with safe_open(checkpoint, framework="pt", device="cpu") as file:
            if file.get_slice("model.diffusion_model.input_blocks.0.0.weight").get_shape()[1] != 4:
                raise ValueError("This version supports 4-channel SDXL text-to-image checkpoints, not inpainting")
        te1, te2, self.vae, self.unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
            sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, checkpoint, "cpu", None)
        # Load without downcasting the VAE, then use bf16 only for the UNet/TEs.
        # Casting bf16 VAE weights back to fp32 would retain lost precision.
        self.text_encoders = [te1, te2]
        self.vae.to(dtype=torch.float32)
        for module in [self.unet, self.vae, *self.text_encoders]:
            module.requires_grad_(False)
            module.eval()
        # Native SDXL and its conditioning are shared with ordinary training.
        train_util.replace_unet_modules(self.unet, False, False, True)
        self.vae.enable_slicing()
        for spec in config["base_loras"]:
            self.base_adapters.append(self._load_base_adapter(spec))
        self.tokenizer = strategy_sdxl.SdxlTokenizeStrategy(config["max_token_length"])
        self.encoder = strategy_sdxl.SdxlTextEncodingStrategy()
        self.unet.to(self.device, dtype=self.dtype)
        for net in self.base_adapters:
            for module in net.unet_loras:
                module.to(self.device, dtype=self.dtype)
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"reference_id": self.identity["reference_id"],
                          "base_adapters": self.adapter_reports}), flush=True)

    def _load_base_adapter(self, spec):
        weights = load_file(spec["path"], device="cpu")
        with safe_open(spec["path"], framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
        module_name = metadata.get("ss_network_module", "networks.lora")
        if module_name == "lycoris.kohya" or any(k.endswith((".w_norm", ".b_norm")) for k in weights):
            from .adapters import reconstruct_lycoris
            network = reconstruct_lycoris(weights, self.text_encoders, self.unet, spec["weight"])
        elif module_name == "networks.lora":
            network, weights = lora.create_network_from_weights(
                spec["weight"], spec["path"], self.vae, self.text_encoders, self.unet,
                weights_sd=weights, for_inference=True)
            network.apply_to(self.text_encoders, self.unet, True, True)
        else:
            raise ValueError(f"Unsupported base adapter module: {module_name}; refusing a partial load")
        # Loading strict=True detects silently omitted normalization, text encoder,
        # scalar or convolution tensors. Do not weaken this to fix incompatibility.
        network.load_state_dict(weights, strict=True)
        network.requires_grad_(False)
        network.eval()
        report = {"path": spec["path"], "weight": spec["weight"], "tensor_count": len(weights),
                  "unet_modules": len(network.unet_loras),
                  "text_encoder_modules": len(network.text_encoder_loras),
                  "normalization_tensors": sum(k.endswith((".w_norm", ".b_norm")) for k in weights),
                  "strict_load": True}
        self.adapter_reports.append(report)
        del weights
        return network

    def attach_preference(self, rank=16, alpha=16, path=None, trainable=False, weight=1.0):
        if self.preference is not None:
            raise RuntimeError("Preference adapter already attached")
        if path:
            with safe_open(path, framework="pt", device="cpu") as file:
                metadata = file.metadata() or {}
            if metadata.get("preference_reference_id") != self.identity["reference_id"]:
                raise ValueError("Preference adapter was trained against a different or unspecified reference")
            if trainable and (float(metadata["ss_network_dim"]) != rank or
                              float(metadata["ss_network_alpha"]) != alpha):
                raise ValueError("Preference adapter rank/alpha do not match the training config")
            weights = load_file(path, device="cpu")
            if any(not key.startswith("lora_unet_") for key in weights):
                raise ValueError("Expected an UNet-only preference adapter")
            net, weights = lora.create_network_from_weights(
                weight, path, self.vae, self.text_encoders, self.unet, weights_sd=weights,
                for_inference=not trainable)
        else:
            net = lora.create_network(weight, rank, alpha, self.vae, self.text_encoders, self.unet)
            weights = None
        net.apply_to(self.text_encoders, self.unet, False, True)
        if weights is not None:
            net.load_state_dict(weights, strict=True)
        net.to(self.device, dtype=torch.float32 if trainable else self.dtype)
        net.requires_grad_(trainable)
        net.train(trainable)
        self.preference = net
        self.preference_weight = weight
        self.unet.set_gradient_checkpointing(trainable)
        self.unet.train(trainable)
        if trainable:
            self.assert_gradient_ownership()
        print(f"Preference adapter: {sum(p.numel() for p in net.parameters()):,} parameters; trainable={trainable}", flush=True)

    def assert_gradient_ownership(self):
        for label, module in [("UNet", self.unet), ("VAE", self.vae),
                              *[(f"text_encoder_{i}", m) for i, m in enumerate(self.text_encoders)],
                              *[(f"base_adapter_{i}", m) for i, m in enumerate(self.base_adapters)]]:
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise AssertionError(f"Frozen component has trainable parameters or gradients: {label}")
        if self.preference is None or not any(p.requires_grad for p in self.preference.parameters()):
            raise AssertionError("No trainable preference parameters")

    @contextmanager
    def reference_mode(self):
        was_training = self.unet.training
        if self.preference is not None:
            self.preference.set_multiplier(0.0)
        self.unet.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            if self.preference is not None:
                self.preference.set_multiplier(self.preference_weight)
            self.unet.train(was_training)

    def _text_device(self, device):
        dtype = self.dtype if device == self.device else torch.float32
        for text_encoder in self.text_encoders:
            text_encoder.to(device, dtype=dtype)
        for network in self.base_adapters:
            for module in network.text_encoder_loras:
                module.to(device, dtype=dtype)

    @torch.no_grad()
    def encode_prompt(self, prompt, width, height):
        key = (prompt, width, height)
        if key not in self.text_cache:
            self._text_device(self.device)
            ids, weights = self.tokenizer.tokenize_with_weights(prompt)
            with torch.autocast("cuda", dtype=self.dtype):
                h1, h2, pooled = self.encoder.encode_tokens_with_weights(
                    self.tokenizer, self.text_encoders, ids, weights)
            size = torch.tensor([[height, width]], device=self.device)
            sizes = sdxl_train_util.get_size_embeddings(size, torch.zeros_like(size), size, self.device)
            text = torch.cat([h1, h2], dim=-1).to(self.dtype)
            vector = torch.cat([pooled, sizes.to(pooled.dtype)], dim=1).to(self.dtype)
            self.text_cache[key] = (text.cpu(), vector.cpu())
            self._text_device(torch.device("cpu"))
            torch.cuda.empty_cache()
        return tuple(t.to(self.device) for t in self.text_cache[key])

    def predict(self, latents, timesteps, text, vector):
        with torch.autocast("cuda", dtype=self.dtype):
            return self.unet(latents.to(self.dtype), timesteps, text.to(self.dtype), vector.to(self.dtype))

    @torch.no_grad()
    def encode_image(self, path, seed):
        image = Image.open(path).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        pixels = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.vae.to(self.device, dtype=torch.float32)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = self.vae.encode(pixels).latent_dist.sample(generator=generator)
        latents = latents * sdxl_model_util.VAE_SCALE_FACTOR
        self.vae.to("cpu")
        if not torch.isfinite(latents).all():
            raise FloatingPointError("VAE produced non-finite latents")
        result = latents.cpu().to(self.dtype)
        del pixels, latents
        torch.cuda.empty_cache()
        return result

    @torch.no_grad()
    def generate(self, prompt, negative_prompt, settings, seed, cancelled=lambda: False):
        width, height = settings["width"], settings["height"]
        text, vector = self.encode_prompt(prompt, width, height)
        use_cfg = settings["cfg"] > 1
        if use_cfg:
            neg_text, neg_vector = self.encode_prompt(negative_prompt, width, height)
            text = torch.cat([neg_text, text])
            vector = torch.cat([neg_vector, vector])
        scheduler_class = {"euler": EulerDiscreteScheduler, "ddim": DDIMScheduler,
                           "dpmpp_2m": DPMSolverMultistepScheduler}[settings["sampler"]]
        scheduler = scheduler_class.from_config(noise_scheduler().config)
        scheduler.set_timesteps(settings["steps"], device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn((1, 4, height // 8, width // 8), generator=generator,
                              device=self.device, dtype=self.dtype) * scheduler.init_noise_sigma
        for index, timestep in enumerate(scheduler.timesteps):
            if cancelled():
                raise Cancelled()
            model_input = torch.cat([latents, latents]) if use_cfg else latents
            model_input = scheduler.scale_model_input(model_input, timestep)
            prediction = self.predict(model_input, timestep.expand(model_input.shape[0]), text, vector)
            if use_cfg:
                uncond, cond = prediction.chunk(2)
                prediction = uncond + settings["cfg"] * (cond - uncond)
            latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
            if (index + 1) % 5 == 0 or index + 1 == settings["steps"]:
                print(f"  seed {seed}: {index + 1}/{settings['steps']} sampling steps", flush=True)
        self.vae.to(self.device, dtype=torch.float32)
        pixels = self.vae.decode(latents.float() / sdxl_model_util.VAE_SCALE_FACTOR).sample
        if not torch.isfinite(pixels).all():
            raise FloatingPointError("VAE decoded non-finite pixels")
        array = (pixels / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()[0]
        self.vae.to("cpu")
        del pixels, latents
        torch.cuda.empty_cache()
        return Image.fromarray((array * 255).round().astype(np.uint8)), dict(scheduler.config)

    def save_preference(self, path, rank, alpha, extra_metadata=None):
        if self.preference is None:
            raise RuntimeError("No preference adapter to save")
        metadata = {"ss_network_module": "networks.lora", "ss_network_dim": str(rank),
                    "ss_network_alpha": str(alpha), "ss_base_model_version": "sdxl_base_v1-0",
                    "ss_network_args": "{}", "preference_reference_id": self.identity["reference_id"],
                    "preference_objective": "diffusion-dpo-v1", "preference_reference": json.dumps(self.identity),
                    **(extra_metadata or {})}
        tensors = {k: v.detach().cpu().float().contiguous() for k, v in self.preference.state_dict().items()}
        save_file(tensors, str(path), metadata=metadata)
