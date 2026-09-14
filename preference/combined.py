"""Export the additive policy as one inference LoRA, preserving all base deltas.

Concatenating factors gives U1@D1 + U2@D2 without cross terms or SVD truncation.
Only plain LoRA/LoCon and additive norm deltas are supported; other algorithms
are rejected explicitly. Computation uses CPU fp32 and never loads the checkpoint.
"""

import json
import math
import os
from pathlib import Path
from uuid import uuid4

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .config import file_identity, model_identity, utc_now


def grouped(weights):
    groups = {}
    for name, tensor in weights.items():
        prefix, dot, suffix = name.partition('.')
        if not dot or not prefix.startswith(('lora_unet_', 'lora_te1_', 'lora_te2_')):
            raise ValueError(f'Unsupported combined-export tensor: {name}')
        groups.setdefault(prefix, {})[suffix] = tensor
    return groups


def scaled_state(state, weight):
    """Normalize each factor group to alpha/rank=1, scaling only the up factor."""
    if not math.isfinite(weight):
        raise ValueError('Adapter weights must be finite')
    if 'w_norm' in state:
        if set(state) - {'w_norm', 'b_norm'}:
            raise ValueError('Unsupported normalization tensor set')
        return {k: v.float() * weight for k, v in state.items()}
    if set(state) - {'lora_up.weight', 'lora_down.weight', 'alpha'} or not {
            'lora_up.weight', 'lora_down.weight'}.issubset(state):
        raise ValueError('Combined export supports plain LoRA/LoCon and norm deltas; DoRA, LoCon mid and other algorithms need a dedicated exporter')
    up, down = state['lora_up.weight'].float(), state['lora_down.weight'].float()
    rank = down.shape[0]
    alpha = float(state.get('alpha', rank))
    if rank < 1 or up.ndim != down.ndim or up.ndim not in (2, 4) or up.shape[1] != rank:
        raise ValueError('Malformed LoRA factor shapes')
    if up.ndim == 4 and up.shape[2:] != (1, 1):
        raise ValueError('Combined export requires a 1x1 convolution up factor')
    # Native LoRA/LyCORIS loaders treat alpha=0 as rank.
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('Invalid LoRA alpha')
    scale = weight * (alpha / rank if alpha else 1.0)
    return {'lora_up.weight': up * scale, 'lora_down.weight': down,
            'alpha': torch.tensor(float(rank))}


def add_state(left, right):
    if 'w_norm' in left and 'w_norm' in right:
        result = dict(left)
        for key, value in right.items():
            if key in result and result[key].shape != value.shape:
                raise ValueError('Normalization shapes differ')
            result[key] = result[key] + value if key in result else value
        return result
    if 'lora_down.weight' not in left or 'lora_down.weight' not in right:
        raise ValueError('Cannot combine different adapter algorithms for the same target')
    a, b = left['lora_down.weight'], right['lora_down.weight']
    u, v = left['lora_up.weight'], right['lora_up.weight']
    if a.shape[1:] != b.shape[1:] or u.shape[0] != v.shape[0] or u.shape[2:] != v.shape[2:]:
        raise ValueError('Cannot concatenate LoRA factors with different target shapes')
    return {'lora_down.weight': torch.cat((a, b), dim=0),
            'lora_up.weight': torch.cat((u, v), dim=1),
            'alpha': torch.tensor(float(a.shape[0] + b.shape[0]))}


def export_combined(config, preference, output, cancelled=lambda: False):
    preference, output = Path(preference).resolve(strict=True), Path(output).absolute()
    if output.suffix.lower() != '.safetensors' or output.exists():
        raise ValueError('Choose a new .safetensors output path; existing files are never overwritten')
    identity = model_identity(config['model'])
    with safe_open(preference, framework='pt', device='cpu') as handle:
        metadata = handle.metadata() or {}
    if metadata.get('preference_reference_id') != identity['reference_id']:
        raise ValueError('Preference adapter belongs to a different reference model/adapter stack')
    if metadata.get('preference_export_type') == 'combined':
        raise ValueError('Select the separate preference_lora checkpoint, not a previously combined export')
    if metadata.get('preference_synthetic_test') == 'true' and not config['training']['allow_synthetic']:
        raise ValueError('Synthetic-test adapter export requires allow_synthetic=true')
    specs = [*config['model']['base_loras'], {'path': str(preference), 'weight': 1.0}]
    merged, sources = {}, []
    for spec in specs:
        if cancelled():
            raise InterruptedError('Combined export stopped before writing')
        print(f"Combining {Path(spec['path']).name} at {spec['weight']}", flush=True)
        sources.append({**file_identity(spec['path']), 'weight': spec['weight']})
        weights = grouped(load_file(spec['path'], device='cpu'))
        for name, state in weights.items():
            if cancelled():
                raise InterruptedError('Combined export stopped before writing')
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError(f'Non-finite adapter tensor in {name}')
            normalized = scaled_state(state, float(spec['weight']))
            merged[name] = add_state(merged[name], normalized) if name in merged else normalized
        del weights
    tensors = {name + '.' + key: value.contiguous() for name, state in merged.items() for key, value in state.items()}
    out_metadata = {k: v for k, v in metadata.items() if k.startswith('preference_')}
    out_metadata.update(ss_network_module='lycoris.kohya', ss_base_model_version='sdxl_base_v1-0',
                        ss_network_dim='Dynamic', ss_network_alpha='Dynamic',
                        ss_network_args=json.dumps({'algo': 'locon', 'train_norm': True}),
                        preference_export_type='combined', preference_exported_at=utc_now(),
                        preference_combination='factor-concatenation-fp32-v1',
                        preference_sources=json.dumps(sources), ss_output_name=output.stem)
    out_metadata['modelspec.title'] = output.stem
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name('.export-' + uuid4().hex + '.tmp')
    try:
        save_file(tensors, str(temp), metadata=out_metadata)
        if cancelled():
            raise InterruptedError('Combined export stopped before publishing')
        # Hard-link publication is atomic and fails if another writer claimed
        # the destination, unlike replace(). Both paths live on the same volume.
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    report = {'combined_lora': str(output), 'bytes': output.stat().st_size,
              'tensor_count': len(tensors), 'source_count': len(specs), 'load_weight': 1.0}
    print(json.dumps(report, indent=2), flush=True)
    return report
