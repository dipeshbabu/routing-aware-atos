"""Hook classes for causal interventions over model residual streams."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RoutedTransportHook:
    """
    Hook for routed transport interventions.

    patch_lookup maps:
        sample_idx -> np.ndarray of shape [seq_len, d_model]

    sample_idx_lookup must align with the batch order used during evaluation.
    """

    def __init__(
        self,
        name: str,
        target_layer: str,
        patch_lookup: dict[int, np.ndarray],
        sample_idx_lookup: list[int],
        target_j_positions: Optional[list[int]] = None,
    ):
        self.name = name
        self.target_layer = target_layer
        self.patch_lookup = patch_lookup
        self.sample_idx_lookup = sample_idx_lookup
        self.target_j_positions = target_j_positions
        self.target_hook_handle = None

    def apply(self, model: Any):
        """Apply target-layer patch hook."""
        target_module = model
        for attr in self.target_layer.split("."):
            target_module = getattr(target_module, attr)

        def routed_transport_hook(module, input_tensors, output):
            try:
                if isinstance(output, torch.Tensor):
                    hidden = output
                    output_is_tuple = False
                elif isinstance(output, tuple):
                    hidden = output[0]
                    output_is_tuple = True
                else:
                    raise RuntimeError(
                        f"Unsupported output type for routed transport: {type(output)}"
                    )

                modified_hidden = hidden.clone()
                batch_size = modified_hidden.shape[0]
                seq_len = modified_hidden.shape[1]

                if batch_size > len(self.sample_idx_lookup):
                    raise RuntimeError(
                        f"Batch size {batch_size} exceeds sample_idx_lookup size {len(self.sample_idx_lookup)}"
                    )

                for batch_pos in range(batch_size):
                    sample_idx = int(self.sample_idx_lookup[batch_pos])
                    if sample_idx not in self.patch_lookup:
                        continue

                    patch_array = self.patch_lookup[sample_idx]
                    if patch_array.ndim != 2:
                        raise RuntimeError(
                            f"Patch array must be 2D [seq_len, d_model], got {patch_array.shape}"
                        )

                    patch_seq_len = min(seq_len, patch_array.shape[0])
                    patch_tensor = torch.from_numpy(
                        patch_array[:patch_seq_len].astype(np.float32)
                    ).to(modified_hidden.device)
                    patch_tensor = patch_tensor.to(modified_hidden.dtype)

                    if self.target_j_positions is None:
                        modified_hidden[batch_pos, :patch_seq_len, :] = patch_tensor
                    else:
                        for j_position in self.target_j_positions:
                            if j_position < patch_seq_len:
                                modified_hidden[batch_pos, j_position, :] = patch_tensor[
                                    j_position
                                ]

                if not output_is_tuple:
                    return modified_hidden

                modified_output = list(output)
                modified_output[0] = modified_hidden
                return tuple(modified_output)

            except Exception as e:
                logger.exception("Routed transport operation failed: %s.", e)
                raise RuntimeError("Routed transport operation failed") from e

        self.target_hook_handle = target_module.register_forward_hook(
            routed_transport_hook
        )
        logger.info(
            "Applied routed transport hook '%s' to target layer '%s'",
            self.name,
            self.target_layer,
        )

    def remove(self):
        """Remove the hook."""
        if self.target_hook_handle:
            self.target_hook_handle.remove()
            self.target_hook_handle = None


class FullSequenceZeroHook:
    """
    Hook that zeros out the full residual stream sequence at a target layer.
    """

    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.hook_handle = None

    def apply(self, model: Any):
        target_module = model
        for attr in self.layer_name.split("."):
            target_module = getattr(target_module, attr)

        def zero_hook(module, input_tensors, output):
            try:
                if isinstance(output, torch.Tensor):
                    return torch.zeros_like(output)
                if isinstance(output, tuple):
                    modified_output = list(output)
                    modified_output[0] = torch.zeros_like(output[0])
                    return tuple(modified_output)
                logger.warning("Unsupported output type: %s", type(output))
                return output
            except Exception as e:
                logger.exception("Full zero operation failed: %s.", e)
                raise RuntimeError("Full zero operation failed") from e

        self.hook_handle = target_module.register_forward_hook(zero_hook)
        logger.info("Applied full-sequence zero hook to layer '%s'", self.layer_name)

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None


def create_routed_transport_hook(
    target_layer: str,
    patch_lookup: dict[int, np.ndarray],
    sample_idx_lookup: list[int],
    j_positions: Optional[list[int]] = None,
) -> RoutedTransportHook:
    """
    Create a routed transport hook that injects precomputed transported residuals
    at the target layer.
    """
    return RoutedTransportHook(
        name="routed_transport_intervention",
        target_layer=target_layer,
        patch_lookup=patch_lookup,
        sample_idx_lookup=sample_idx_lookup,
        target_j_positions=j_positions,
    )


def create_routed_transport_hook_family(
    target_layer: str,
    patch_lookup: dict[int, np.ndarray],
    sample_idx_lookup: list[int],
    js: list[list[int]],
    prefix: str,
) -> dict[str, RoutedTransportHook]:
    """
    Create a family of routed transport hooks for different target position sets.
    """
    hooks = {}
    for j in js:
        hooks[f"{prefix}_{str(j)}"] = create_routed_transport_hook(
            target_layer=target_layer,
            patch_lookup=patch_lookup,
            sample_idx_lookup=sample_idx_lookup,
            j_positions=j,
        )
    return hooks


def create_full_sequence_zero_hook(layer_name: str) -> FullSequenceZeroHook:
    """Create a hook that zeros the full sequence residual stream at a target layer."""
    return FullSequenceZeroHook(layer_name)
