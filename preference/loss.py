"""Diffusion-DPO's pairwise denoising objective; see preference/README.md."""

import torch
import torch.nn.functional as F


def diffusion_dpo_loss(policy_errors, reference_errors, beta, weights=None):
    """Errors are [pairs, 2] in winner/rejected order, averaged per image.

    The -beta/2 convention follows SalesforceAIResearch/DiffusionDPO and the
    Diffusers SDXL LoRA example. Strength scales each pair's contribution;
    dividing by pair count (not weight sum) preserves weights at microbatch 1.
    """
    if policy_errors.ndim != 2 or policy_errors.shape[1] != 2 or reference_errors.shape != policy_errors.shape:
        raise ValueError("Expected equally shaped [pair_count, 2] error tensors")
    if policy_errors.shape[0] == 0 or beta <= 0:
        raise ValueError("A positive beta and at least one comparison are required")
    policy = policy_errors.float()
    reference = reference_errors.detach().float()
    margin = (policy[:, 0] - policy[:, 1]) - (reference[:, 0] - reference[:, 1])
    logits = -0.5 * beta * margin
    per_pair = -F.logsigmoid(logits)
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=per_pair.dtype, device=per_pair.device)
        if weights.shape != per_pair.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Expected one finite positive strength weight per pair")
        per_pair = per_pair * weights
    if not torch.isfinite(per_pair).all():
        raise FloatingPointError("Non-finite DPO loss; no optimizer update should be applied")
    return per_pair.mean(), {
        "logit": logits.detach().mean().item(),
        "preference_accuracy": ((logits > 0).float() + 0.5 * (logits == 0).float()).mean().item(),
        "policy_winner_mse": policy[:, 0].detach().mean().item(),
        "policy_rejected_mse": policy[:, 1].detach().mean().item(),
        "reference_winner_mse": reference[:, 0].mean().item(),
        "reference_rejected_mse": reference[:, 1].mean().item(),
    }
