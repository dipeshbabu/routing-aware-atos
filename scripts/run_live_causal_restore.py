from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch

from routing_aware_atos.activation_loader import ActivationLoader
from routing_aware_atos.causal_eval.live_restore import evaluate_live_causal_restoration
from routing_aware_atos.models.transport_operator import TransportOperator
from routing_aware_atos.provenance import (
    fingerprint_matches,
    live_causal_run_provenance,
    sha256_payload,
)
from routing_aware_atos.routed_dataset import build_concatenated_routed_pairs, build_routed_pairs
from routing_aware_atos.routing_policies import build_routing_policy
from routing_aware_atos.utils.io import load_json, load_yaml, save_json


def _resolve_dtype(name: str) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def _build_patch(
    sample,
    *,
    operator: TransportOperator,
    policy,
    source_layer: int,
    target_layer: int,
    input_mode: str,
    max_sources: int,
) -> np.ndarray:
    if input_mode == "concat":
        pairs = build_concatenated_routed_pairs(
            [sample],
            source_layer=source_layer,
            target_layer=target_layer,
            routing_policy=policy,
            max_sources=max_sources,
        )
    elif input_mode == "weighted_sum":
        pairs = build_routed_pairs(
            [sample],
            source_layer=source_layer,
            target_layer=target_layer,
            routing_policy=policy,
        )
    else:
        raise ValueError(f"Unknown input_mode {input_mode!r}")
    patch = operator.predict(pairs.X).astype(np.float32, copy=False)
    if patch.shape[0] != sample.seq_len:
        raise RuntimeError(f"Expected one patch per sequence position, got {patch.shape[0]} for {sample.seq_len}")
    return patch


def _position_lookup(
    sample_indices: list[int],
    attention_masks: np.ndarray,
    *,
    count: int,
    seed: int,
) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    lookup: dict[int, list[int]] = {}
    for row, sample_idx in enumerate(sample_indices):
        valid_length = int(attention_masks[row].sum())
        eligible = np.arange(1, max(1, valid_length - 1), dtype=np.int64)
        if eligible.size < count:
            raise ValueError(
                f"Sequence {sample_idx} has only {eligible.size} causal positions, requested {count}"
            )
        lookup[sample_idx] = sorted(rng.choice(eligible, size=count, replace=False).astype(int).tolist())
    return lookup


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _summarize_repeats(rows: list[dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_repeats": len(rows)}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    prefixes = ["clean", "ablated", "restored"]
    if "null_cross_entropy" in summary:
        prefixes.append("null")
    for prefix in prefixes:
        ce_key = f"{prefix}_cross_entropy"
        summary[f"{prefix}_perplexity"] = float(math.exp(summary[ce_key]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Gemma causal perplexity restoration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--operator-path")
    parser.add_argument("--source-layer", type=int)
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--policy")
    parser.add_argument("--output-path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root_cfg = load_yaml(args.config)
    cfg = dict(root_cfg.get("live_causal", root_cfg))

    model_name = str(cfg.get("model_name", root_cfg.get("model_name")))
    activation_dir = str(
        cfg.get("activation_dir_path")
        or root_cfg.get("collection", {}).get("output_dir")
    )
    operator_path = args.operator_path or cfg.get("operator_path")
    source_layer = args.source_layer if args.source_layer is not None else int(cfg["source_layer"])
    target_layer = args.target_layer if args.target_layer is not None else int(cfg["target_layer"])
    policy_name = args.policy or str(cfg.get("routing_policy", "same_token"))
    output_path = args.output_path or cfg.get("output_path")
    if not operator_path or not output_path:
        raise ValueError("operator_path and output_path are required")

    fingerprint_inputs = live_causal_run_provenance(
        root_cfg,
        operator_path=operator_path,
        source_layer=source_layer,
        target_layer=target_layer,
        policy=policy_name,
    )
    run_fingerprint = sha256_payload(fingerprint_inputs)
    output_file = Path(output_path)
    if output_file.exists() and not args.force:
        try:
            existing = load_json(output_file)
        except (OSError, ValueError):
            existing = {}
        if fingerprint_matches(existing, expected_provenance=fingerprint_inputs):
            print(f"SKIP: current live causal result already exists -> {output_file}")
            return
        print(f"Recomputing stale live causal result -> {output_file}")

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional real-model dependency
        raise ImportError("Install the real-model extra for live causal evaluation") from exc

    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_kwargs: dict[str, Any] = {
        "dtype": _resolve_dtype(str(cfg.get("dtype", "bfloat16"))),
        "revision": root_cfg.get("model_revision"),
        "token": cfg.get("token"),
    }
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()

    operator = TransportOperator.load(operator_path)
    top_k = int(cfg.get("top_k", 3))
    policy = build_routing_policy(
        policy_name,
        top_k=top_k,
        normalize_weights=bool(cfg.get("normalize_weights", True)),
        exclude_self=bool(cfg.get("exclude_self", False)),
        allow_negative_scores=bool(cfg.get("allow_negative_scores", False)),
        random_seed=int(cfg.get("random_seed", 0)),
        causal_only=bool(cfg.get("causal_only", True)),
    )
    input_mode = str(cfg.get("input_mode", "weighted_sum"))
    max_sources = int(cfg.get("max_sources", top_k))

    patch_lookup: dict[int, np.ndarray] = {}
    null_patch_lookup: dict[int, np.ndarray] = {}
    input_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    with ActivationLoader(activation_dir_path=activation_dir) as loader:
        causal_indices = loader.indices_for_split(str(cfg.get("split_name", "causal")))
        causal_indices = causal_indices[: int(cfg.get("num_sequences", 100))]
        for sample_idx in causal_indices:
            sample = loader.get_cached_sample(
                sample_idx,
                layer_indices=[source_layer, target_layer],
                attention_layer_pairs=(
                    [(source_layer, target_layer)] if policy.requires_attention else None
                ),
                attribution_layer_pairs=(
                    [(source_layer, target_layer)] if policy.requires_attribution else None
                ),
            )
            patch_lookup[sample_idx] = _build_patch(
                sample,
                operator=operator,
                policy=policy,
                source_layer=source_layer,
                target_layer=target_layer,
                input_mode=input_mode,
                max_sources=max_sources,
            )
            null_patch_lookup[sample_idx] = operator.predict(
                np.zeros((sample.seq_len, operator.weight.shape[0]), dtype=np.float32)
            ).astype(np.float32, copy=False)
            input_rows.append(loader.get_input_ids(sample_idx))
            mask_rows.append(loader.get_attention_mask(sample_idx))

    input_ids = np.stack(input_rows).astype(np.int64)
    attention_masks = np.stack(mask_rows).astype(np.int64)
    batch_size = int(cfg.get("batch_size", 2))
    repeats = int(cfg.get("position_repeats", 3))
    seed = int(cfg.get("position_seed", 17))
    position_counts = cfg.get("position_counts", [1, 5, "all"])
    target_module = str(cfg.get("target_module", f"model.layers.{target_layer}"))
    results: dict[str, Any] = {}

    for raw_count in position_counts:
        is_all = str(raw_count).lower() == "all"
        mode_name = "all" if is_all else str(int(raw_count))
        repeat_rows: list[dict[str, float]] = []
        repeat_count = 1 if is_all else repeats
        for repeat_idx in range(repeat_count):
            positions = None
            if not is_all:
                positions = _position_lookup(
                    causal_indices,
                    attention_masks,
                    count=int(raw_count),
                    seed=seed + repeat_idx,
                )
            batch_rows: list[dict[str, float]] = []
            for start in range(0, len(causal_indices), batch_size):
                stop = min(start + batch_size, len(causal_indices))
                batch_indices = causal_indices[start:stop]
                batch_positions = (
                    None
                    if positions is None
                    else {sample_idx: positions[sample_idx] for sample_idx in batch_indices}
                )
                batch_rows.append(
                    evaluate_live_causal_restoration(
                        model,
                        torch.from_numpy(input_ids[start:stop]),
                        attention_mask=torch.from_numpy(attention_masks[start:stop]),
                        target_layer=target_module,
                        patch_lookup=patch_lookup,
                        sample_idx_lookup=batch_indices,
                        position_lookup=batch_positions,
                        null_patch_lookup=null_patch_lookup,
                    )
                )
            repeat_rows.append(_aggregate(batch_rows))
        results[mode_name] = {
            "summary": _summarize_repeats(repeat_rows),
            "repeats": repeat_rows,
        }
        print(f"Completed live causal mode positions={mode_name}")

    payload = {
        "model_name": model_name,
        "activation_dir_path": activation_dir,
        "operator_path": str(operator_path),
        "source_layer": source_layer,
        "target_layer": target_layer,
        "target_module": target_module,
        "routing_policy": policy_name,
        "input_mode": input_mode,
        "causal_only": bool(policy.config.causal_only),
        "model_revision": root_cfg.get("model_revision"),
        "run_fingerprint": run_fingerprint,
        "provenance": fingerprint_inputs,
        "num_sequences": len(causal_indices),
        "sequence_length": int(input_ids.shape[1]),
        "position_repeats": repeats,
        "results": results,
    }
    save_json(output_path, payload)
    print(f"Saved live causal restoration results -> {output_path}")


if __name__ == "__main__":
    main()
