"""Strict native LoRA/LoCon reconstruction, including LyCORIS norm deltas.

Some LyCORIS versions list optional PiSSA tensors but do not accept them in
LoConModule.make_module_from_state_dict. This explicit supported-key mapping
avoids that factory mismatch without patching LyCORIS or dropping tensors.
"""

import torch


class FrozenAdapters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet_loras = []
        self.text_encoder_loras = []


@torch.no_grad()
def reconstruct_lycoris(weights, text_encoders, unet, multiplier):
    from lycoris.modules.locon import LoConModule
    from lycoris.modules.norms import NormModule

    targets = {"lora_unet_" + name.replace(".", "_"): (module, False)
               for name, module in unet.named_modules()}
    for index, encoder in enumerate(text_encoders, 1):
        targets.update({f"lora_te{index}_" + name.replace(".", "_"): (module, True)
                        for name, module in encoder.named_modules()})
    grouped = {}
    for name, tensor in weights.items():
        if "." not in name:
            raise ValueError(f"Unsupported adapter tensor: {name}")
        prefix, suffix = name.split(".", 1)
        grouped.setdefault(prefix, {})[suffix] = tensor
    network = FrozenAdapters()
    for name, state in grouped.items():
        if name not in targets:
            raise ValueError(f"Adapter target does not exist in this checkpoint architecture: {name}")
        target, is_text = targets[name]
        if "lora_up.weight" in state:
            allowed = {"lora_up.weight", "lora_down.weight", "lora_mid.weight", "alpha", "dora_scale"}
            required = {"lora_up.weight", "lora_down.weight", "alpha"}
            if set(state) - allowed or not required.issubset(state):
                raise ValueError(f"Unsupported/incomplete LoCon tensor set for {name}: {sorted(state)}")
            module = LoConModule.make_module_from_state_dict(
                name, target, state["lora_up.weight"], state["lora_down.weight"],
                state.get("lora_mid.weight"), state["alpha"], state.get("dora_scale"))
        elif "w_norm" in state:
            if set(state) - {"w_norm", "b_norm"}:
                raise ValueError(f"Unsupported normalization tensors in {name}")
            module = NormModule.make_module_from_state_dict(name, target, state["w_norm"], state.get("b_norm"))
        else:
            raise ValueError(f"Unsupported LyCORIS algorithm in {name}; only LoRA/LoCon and norm deltas are supported")
        module.multiplier = multiplier
        module.load_state_dict(state, strict=True)
        module.apply_to()
        network.add_module(name, module)
        (network.text_encoder_loras if is_text else network.unet_loras).append(module)
    if set(network.state_dict()) != set(weights):
        raise ValueError("Reconstructed adapter does not account for every saved tensor")
    network.requires_grad_(False)
    network.eval()
    return network
