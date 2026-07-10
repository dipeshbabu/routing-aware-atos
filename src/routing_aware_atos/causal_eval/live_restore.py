from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from routing_aware_atos.causal_eval.hooks import create_full_sequence_zero_hook, create_routed_transport_hook


def _extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, tuple) and model_output:
        return model_output[0]
    raise TypeError(f"Cannot extract logits from output type {type(model_output)}")


def next_token_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(f"logits and input_ids sequence shapes differ: {tuple(logits.shape[:2])} vs {tuple(input_ids.shape)}")
    if input_ids.shape[1] < 2:
        return 0.0
    pred = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    return float(F.cross_entropy(pred.view(-1, pred.shape[-1]), labels.view(-1)).detach().cpu())


def mean_token_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(f"logit shapes differ: {tuple(reference_logits.shape)} vs {tuple(candidate_logits.shape)}")
    ref_log_probs = F.log_softmax(reference_logits, dim=-1)
    cand_log_probs = F.log_softmax(candidate_logits, dim=-1)
    ref_probs = ref_log_probs.exp()
    kl = ref_probs * (ref_log_probs - cand_log_probs)
    return float(kl.sum(dim=-1).mean().detach().cpu())


def logit_mse(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(f"logit shapes differ: {tuple(reference_logits.shape)} vs {tuple(candidate_logits.shape)}")
    return float(torch.mean((reference_logits - candidate_logits) ** 2).detach().cpu())


def restoration_fraction(restored_error: float, ablated_error: float) -> float:
    if ablated_error <= 1e-12:
        return 0.0
    return float(1.0 - restored_error / ablated_error)


def evaluate_live_causal_restoration(
    model: Any,
    input_ids: torch.Tensor,
    *,
    target_layer: str,
    patch_lookup: Dict[int, np.ndarray],
    sample_idx_lookup: list[int],
) -> Dict[str, float]:
    """
    Run clean, zero-ablated, and ATO-restored forwards through a model.

    The target layer must emit a tensor shaped [batch, seq, d_model], or a tuple
    whose first element has that shape. patch_lookup maps sample_idx to a full
    sequence of transported residuals [seq, d_model].
    """
    model_was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = input_ids.device
    input_ids = input_ids.to(device)

    try:
        with torch.no_grad():
            clean_logits = _extract_logits(model(input_ids)).detach()

            zero_hook = create_full_sequence_zero_hook(target_layer)
            zero_hook.apply(model)
            try:
                ablated_logits = _extract_logits(model(input_ids)).detach()
            finally:
                zero_hook.remove()

            restore_hook = create_routed_transport_hook(
                target_layer=target_layer,
                patch_lookup=patch_lookup,
                sample_idx_lookup=sample_idx_lookup,
            )
            restore_hook.apply(model)
            try:
                restored_logits = _extract_logits(model(input_ids)).detach()
            finally:
                restore_hook.remove()
    finally:
        if model_was_training:
            model.train()

    ablated_kl = mean_token_kl(clean_logits, ablated_logits)
    restored_kl = mean_token_kl(clean_logits, restored_logits)
    ablated_mse = logit_mse(clean_logits, ablated_logits)
    restored_mse = logit_mse(clean_logits, restored_logits)
    clean_ce = next_token_cross_entropy(clean_logits, input_ids)
    ablated_ce = next_token_cross_entropy(ablated_logits, input_ids)
    restored_ce = next_token_cross_entropy(restored_logits, input_ids)

    return {
        "clean_cross_entropy": clean_ce,
        "ablated_cross_entropy": ablated_ce,
        "restored_cross_entropy": restored_ce,
        "ablated_kl": ablated_kl,
        "restored_kl": restored_kl,
        "kl_restoration": restoration_fraction(restored_kl, ablated_kl),
        "ablated_logit_mse": ablated_mse,
        "restored_logit_mse": restored_mse,
        "logit_mse_restoration": restoration_fraction(restored_mse, ablated_mse),
        "cross_entropy_delta_restoration": restoration_fraction(
            restored_ce - clean_ce,
            ablated_ce - clean_ce,
        ),
    }
