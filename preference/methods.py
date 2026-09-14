"""Torch-free method settings. Legacy DPO settings and resume hashes stay valid."""

from .config import finite_number

ADDIFT = {"min_timestep": 400, "max_timestep": 900,
          "alternate_inverse": True}
LECO = {"target": "", "positive": "", "neutral": "", "unconditional": "",
        "action": "enhance", "guidance_scale": 1.0, "denoising_steps": 20,
        "denoise_cfg": 3.0}


def method_settings(training):
    method = training.get("objective", "dpo")
    if method not in ("dpo", "addift", "leco"):
        raise ValueError("training.objective must be dpo, addift, or leco")
    defaults = ADDIFT if method == "addift" else LECO if method == "leco" else {}
    raw = training.get(method, {}) if method != "dpo" else {}
    if not isinstance(raw, dict) or set(raw) - defaults.keys():
        raise ValueError(f"Unknown or invalid {method} settings")
    values = {**defaults, **raw}
    if method == "addift":
        lo, hi = values["min_timestep"], values["max_timestep"]
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= 1000:
            raise ValueError("ADDifT requires 0 <= min_timestep < max_timestep <= 1000 (exclusive)")
        if type(values["alternate_inverse"]) is not bool:
            raise ValueError("alternate_inverse must be true or false")
    if method == "leco":
        for name in ("target", "positive", "neutral", "unconditional"):
            if not isinstance(values[name], str):
                raise ValueError(f"LECO {name} must be text")
        if not values["target"].strip() or not values["positive"].strip():
            raise ValueError("Enter LECO's prompt to change and concept to enhance/erase")
        if values["positive"] == values["unconditional"]:
            raise ValueError("LECO concept and contrast baseline must differ")
        if values["action"] not in ("enhance", "erase"):
            raise ValueError("LECO action must be enhance or erase")
        for name in ("guidance_scale", "denoise_cfg"):
            values[name] = finite_number(values[name], f"LECO {name}", positive=True)
        if values["denoise_cfg"] < 1:
            raise ValueError("LECO denoise_cfg must be at least 1")
        steps = values["denoising_steps"]
        if type(steps) is not int or not 2 <= steps <= 100:
            raise ValueError("LECO denoising_steps must be an integer from 2 to 100")
    return method, values


def signature_settings(training, generation):
    method, values = method_settings(training)
    result = {k: v for k, v in training.items()
              if k not in ("max_steps", "checkpoint_every", "output_dir", "objective", "addift", "leco")}
    if method != "dpo":
        result.update(objective=method, **{method: values})
    if method == "leco":
        result["resolution"] = [generation["width"], generation["height"]]
    if any(key in training for key in ("optimizer", "lr_schedule", "max_grad_norm")):
        from .optimization import optimization_settings
        for key in ("optimizer", "lr_schedule", "max_grad_norm"):
            result.pop(key, None)
        result["optimization"] = optimization_settings(training)
    return result
