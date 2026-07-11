"""Hook classes for causal interventions over model residual streams."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)
PositionSelection = list[int] | Mapping[int, list[int]] | None


def _positions_for_sample(selection: PositionSelection, sample_idx: int) -> list[int] | None:
    if selection is None:
        return None
    if isinstance(selection, Mapping):
        if sample_idx not in selection:
            raise KeyError(f"Missing position selection for sample {sample_idx}")
        return selection[sample_idx]
    return selection


class RoutedTransportHook:
    """Patch target-layer residuals with precomputed transported vectors."""

    def __init__(
        self,
        name: str,
        target_layer: str,
        patch_lookup: dict[int, np.ndarray],
        sample_idx_lookup: list[int],
        target_j_positions: PositionSelection = None,
    ):
        self.name = name
        self.target_layer = target_layer
        self.patch_lookup = patch_lookup
        self.sample_idx_lookup = sample_idx_lookup
        self.target_j_positions = target_j_positions
        self.target_hook_handle = None

    def apply(self, model: Any):
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
                    raise RuntimeError(f"Unsupported output type for routed transport: {type(output)}")

                modified_hidden = hidden.clone()
                if modified_hidden.shape[0] != len(self.sample_idx_lookup):
                    raise RuntimeError(
                        f"Batch size {modified_hidden.shape[0]} does not match sample_idx_lookup size "
                        f"{len(self.sample_idx_lookup)}"
                    )
                for batch_pos in range(modified_hidden.shape[0]):
                    sample_idx = int(self.sample_idx_lookup[batch_pos])
                    if sample_idx not in self.patch_lookup:
                        raise KeyError(f"Missing transport patch for sample {sample_idx}")
                    patch_array = np.asarray(self.patch_lookup[sample_idx], dtype=np.float32)
                    if patch_array.ndim != 2:
                        raise RuntimeError(f"Patch array must be 2D [seq_len, d_model], got {patch_array.shape}")
                    if patch_array.shape[1] != modified_hidden.shape[2]:
                        raise RuntimeError(
                            f"Patch width {patch_array.shape[1]} does not match hidden width "
                            f"{modified_hidden.shape[2]}"
                        )
                    patch_seq_len = min(modified_hidden.shape[1], patch_array.shape[0])
                    patch_tensor = torch.from_numpy(patch_array[:patch_seq_len]).to(
                        device=modified_hidden.device,
                        dtype=modified_hidden.dtype,
                    )
                    positions = _positions_for_sample(self.target_j_positions, sample_idx)
                    if positions is None:
                        modified_hidden[batch_pos, :patch_seq_len, :] = patch_tensor
                    else:
                        for position in positions:
                            if position < 0 or position >= patch_seq_len:
                                raise IndexError(
                                    f"Patch position {position} is outside [0, {patch_seq_len - 1}]"
                                )
                            modified_hidden[batch_pos, position, :] = patch_tensor[position]

                if not output_is_tuple:
                    return modified_hidden
                modified_output = list(output)
                modified_output[0] = modified_hidden
                return tuple(modified_output)
            except Exception as exc:
                logger.exception("Routed transport operation failed: %s", exc)
                raise RuntimeError("Routed transport operation failed") from exc

        self.target_hook_handle = target_module.register_forward_hook(routed_transport_hook)

    def remove(self):
        if self.target_hook_handle:
            self.target_hook_handle.remove()
            self.target_hook_handle = None


class FullSequenceZeroHook:
    """Zero all or selected target-layer residual positions."""

    def __init__(
        self,
        layer_name: str,
        *,
        sample_idx_lookup: list[int] | None = None,
        target_j_positions: PositionSelection = None,
    ):
        self.layer_name = layer_name
        self.sample_idx_lookup = sample_idx_lookup
        self.target_j_positions = target_j_positions
        self.hook_handle = None

    def apply(self, model: Any):
        target_module = model
        for attr in self.layer_name.split("."):
            target_module = getattr(target_module, attr)

        def zero_hook(module, input_tensors, output):
            try:
                if isinstance(output, torch.Tensor):
                    hidden = output
                    output_is_tuple = False
                elif isinstance(output, tuple):
                    hidden = output[0]
                    output_is_tuple = True
                else:
                    logger.warning("Unsupported output type: %s", type(output))
                    return output

                if self.target_j_positions is None:
                    modified_hidden = torch.zeros_like(hidden)
                else:
                    if self.sample_idx_lookup is None:
                        raise RuntimeError("sample_idx_lookup is required for position-specific zeroing")
                    if hidden.shape[0] != len(self.sample_idx_lookup):
                        raise RuntimeError(
                            f"Batch size {hidden.shape[0]} does not match sample_idx_lookup size "
                            f"{len(self.sample_idx_lookup)}"
                        )
                    modified_hidden = hidden.clone()
                    for batch_pos in range(hidden.shape[0]):
                        sample_idx = int(self.sample_idx_lookup[batch_pos])
                        positions = _positions_for_sample(self.target_j_positions, sample_idx) or []
                        for position in positions:
                            if position < 0 or position >= hidden.shape[1]:
                                raise IndexError(
                                    f"Zero position {position} is outside [0, {hidden.shape[1] - 1}]"
                                )
                            modified_hidden[batch_pos, position, :] = 0

                if not output_is_tuple:
                    return modified_hidden
                modified_output = list(output)
                modified_output[0] = modified_hidden
                return tuple(modified_output)
            except Exception as exc:
                logger.exception("Zero operation failed: %s", exc)
                raise RuntimeError("Zero operation failed") from exc

        self.hook_handle = target_module.register_forward_hook(zero_hook)

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None


def create_routed_transport_hook(
    target_layer: str,
    patch_lookup: dict[int, np.ndarray],
    sample_idx_lookup: list[int],
    j_positions: PositionSelection = None,
) -> RoutedTransportHook:
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
    return {
        f"{prefix}_{str(positions)}": create_routed_transport_hook(
            target_layer=target_layer,
            patch_lookup=patch_lookup,
            sample_idx_lookup=sample_idx_lookup,
            j_positions=positions,
        )
        for positions in js
    }


def create_full_sequence_zero_hook(
    layer_name: str,
    *,
    sample_idx_lookup: list[int] | None = None,
    j_positions: PositionSelection = None,
) -> FullSequenceZeroHook:
    return FullSequenceZeroHook(
        layer_name,
        sample_idx_lookup=sample_idx_lookup,
        target_j_positions=j_positions,
    )
