"""Optimization spec + native builder (torch-free import)."""
import math

_OPT_TYPES = ("AdamW", "SimplifiedAdEMAMixExM")
_ADAMW_ARGS = {"betas", "eps", "weight_decay", "amsgrad"}
_CUSTOM_ARGS = {
    "betas", "min_beta1", "beta1_warmup", "alpha", "weight_decay", "eps",
    "amsgrad_min_decay_rate", "amsgrad_max_decay_rate", "use_adabelief",
    "torch_compile", "update_strategy", "use_orthograd", "use_compass",
    "use_newton_schulz", "use_stable_spam_clipping", "state_storage_dtype",
    "state_storage_device",
}
_UPDATE_STRATEGIES = ("unmodified", "cautious", "grams", "both")
_DTYPES = ("float32", "float16", "bfloat16")
_DEVICES = ("cpu", "cuda")


def _num(v, name, lo=None, hi=None, lo_open=False, hi_open=False, default=None):
    if v is None and default is not None:
        v = default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(v).__name__}")
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise ValueError(f"{name} below lower bound")
    if hi is not None and (v >= hi if hi_open else v > hi):
        raise ValueError(f"{name} above upper bound")
    return v


def _bool(v, name):
    if not isinstance(v, bool):
        raise ValueError(f"{name} must be bool")
    return v


def _choice(v, name, allowed):
    if v not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {v!r}")
    return v


def _betas(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return [_num(x, "optimizer.args.betas", 0.0, 1.0, hi_open=True) for x in v]
    raise ValueError("betas must be a two-element sequence")


def _reject_extra(d, allowed, name):
    extra = set(d) - set(allowed)
    if extra:
        raise ValueError(f"unknown {name} keys: {sorted(extra)}")


def _optimizer_settings(training, max_steps):
    raw = training.get("optimizer", {})
    if not isinstance(raw, dict):
        raise ValueError("training.optimizer must be a dict")
    _reject_extra(raw, ("type", "args"), "optimizer")
    typ = _choice(raw.get("type", "AdamW"), "optimizer.type", _OPT_TYPES)
    args_in = raw.get("args", {})
    if not isinstance(args_in, dict):
        raise ValueError("training.optimizer.args must be a dict")
    _reject_extra(args_in, _ADAMW_ARGS if typ == "AdamW" else _CUSTOM_ARGS,
                  "optimizer.args")
    args, horizon = {}, None
    if "betas" in args_in:
        args["betas"] = _betas(args_in["betas"])
    if "eps" in args_in:
        args["eps"] = _num(args_in["eps"], "optimizer.args.eps", lo=0.0)
    args["weight_decay"] = _num(args_in.get("weight_decay", 0.0),
                                "optimizer.args.weight_decay", lo=0.0)
    if typ == "AdamW":
        if "amsgrad" in args_in:
            args["amsgrad"] = _bool(args_in["amsgrad"], "optimizer.args.amsgrad")
        return typ, args, horizon
    if "min_beta1" in args_in:
        args["min_beta1"] = _num(args_in["min_beta1"], "min_beta1", 0.0, 1.0, hi_open=True)
    if "alpha" in args_in:
        args["alpha"] = _num(args_in["alpha"], "alpha", lo=0.0)
    if "amsgrad_min_decay_rate" in args_in:
        args["amsgrad_min_decay_rate"] = _num(args_in["amsgrad_min_decay_rate"],
                                              "amsgrad_min_decay_rate", 0.0, 1.0)
    if "amsgrad_max_decay_rate" in args_in:
        args["amsgrad_max_decay_rate"] = _num(args_in["amsgrad_max_decay_rate"],
                                              "amsgrad_max_decay_rate", 0.0, 1.0)
    for flag in ("use_adabelief", "use_orthograd", "use_compass",
                 "use_newton_schulz", "use_stable_spam_clipping"):
        if flag in args_in:
            args[flag] = _bool(args_in[flag], flag)
    if "update_strategy" in args_in:
        args["update_strategy"] = _choice(args_in["update_strategy"],
                                          "update_strategy", _UPDATE_STRATEGIES)
    if "state_storage_dtype" in args_in:
        args["state_storage_dtype"] = _choice(args_in["state_storage_dtype"],
                                              "state_storage_dtype", _DTYPES)
    if "state_storage_device" in args_in:
        args["state_storage_device"] = _choice(args_in["state_storage_device"],
                                               "state_storage_device", _DEVICES)
    if "beta1_warmup" in args_in:
        bw = args_in["beta1_warmup"]
        if bw is None:
            args["beta1_warmup"] = None
        elif isinstance(bw, str):
            if bw != "total_steps":
                raise ValueError("beta1_warmup string must be 'total_steps'")
            args["beta1_warmup"] = max_steps
            horizon = max_steps
        elif isinstance(bw, bool) or not isinstance(bw, int):
            raise ValueError("beta1_warmup must be None, int, or 'total_steps'")
        elif not 1 <= bw <= max_steps:
            raise ValueError("beta1_warmup int must be in [1, max_steps]")
        else:
            args["beta1_warmup"] = bw
    args["torch_compile"] = _bool(args_in.get("torch_compile", False), "torch_compile")
    return typ, args, horizon


def _schedule_settings(training, max_steps, lr):
    raw = training.get("lr_schedule", {})
    if not isinstance(raw, dict):
        raise ValueError("training.lr_schedule must be a dict")
    typ = raw.get("type", "constant")
    if typ == "constant":
        _reject_extra(raw, ("type",), "lr_schedule")
        return {"type": "constant"}, None
    if typ != "rawr":
        raise ValueError(f"lr_schedule.type must be constant or rawr, got {typ!r}")
    _reject_extra(raw, ("type", "warmup_ratio", "min_lr", "gamma", "d"), "lr_schedule")
    wr = _num(raw.get("warmup_ratio", 0.05), "warmup_ratio", 0.0, 1.0, hi_open=True)
    min_lr = _num(raw.get("min_lr", 1e-6), "min_lr", 0.0, lr)
    gamma = _num(raw.get("gamma", 0.9), "gamma", 0.0, 1.0, lo_open=True)
    d = _num(raw.get("d", 0.9), "d", 0.0, 1.0, hi_open=True)
    warmup_steps = int(math.ceil(max_steps * wr))
    if warmup_steps >= max_steps:
        raise ValueError("warmup_steps must be < max_steps")
    return {"type": "rawr", "first_cycle_max_steps": max_steps,
            "warmup_steps": warmup_steps, "min_lr": min_lr,
            "gamma": gamma, "d": d}, max_steps


def optimization_settings(training):
    """Validate training config -> new normalized spec dict (input untouched)."""
    if not isinstance(training, dict):
        raise ValueError("training must be a dict")
    lr = _num(training.get("learning_rate", 1e-5), "learning_rate", lo=0.0, lo_open=True)
    ms = training.get("max_steps", 100)
    if isinstance(ms, bool) or not isinstance(ms, int) or ms <= 0:
        raise ValueError("max_steps must be a positive int")
    opt_type, args, opt_horizon = _optimizer_settings(training, ms)
    sched, sched_horizon = _schedule_settings(training, ms, lr)
    out = {
        "optimizer": {"type": opt_type, "args": args},
        "lr_schedule": sched,
        "max_grad_norm": _num(training.get("max_grad_norm", 1.0), "max_grad_norm", lo=0.0),
    }
    horizon = opt_horizon if opt_horizon is not None else sched_horizon
    if horizon is not None:
        out["horizon_steps"] = horizon
    return out


def build_optimization(parameters, training):
    """Build (optimizer, scheduler_or_None, spec) from training config."""
    spec = optimization_settings(training)
    lr = float(training.get("learning_rate", 1e-5))
    args = dict(spec["optimizer"]["args"])
    if "betas" in args:
        args["betas"] = tuple(args["betas"])
    if spec["optimizer"]["type"] == "AdamW":
        from torch.optim import AdamW
        optimizer = AdamW(parameters, lr=lr, **args)
    else:
        from LoraEasyCustomOptimizer.ademamix import SimplifiedAdEMAMixExM
        optimizer = SimplifiedAdEMAMixExM(parameters, lr=lr, **args)
    scheduler = None
    sched = spec["lr_schedule"]
    if sched["type"] == "rawr":
        from LoraEasyCustomOptimizer.RexAnnealingWarmRestarts import RexAnnealingWarmRestarts
        kwargs = {k: v for k, v in sched.items() if k != "type"}
        scheduler = RexAnnealingWarmRestarts(optimizer, cycle_multiplier=1, **kwargs)
    return optimizer, scheduler, spec
