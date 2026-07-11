from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from routing_aware_atos.activation_loader import ActivationLoader
from routing_aware_atos.provenance import (
    feature_selection_implementation_sha256,
    sha256_file,
)
from routing_aware_atos.sae.encoding import validate_sae_artifact
from routing_aware_atos.sae.feature_selection import (
    binned_entropy,
    composite_feature_scores,
    max_abs_anchor_correlation,
    normalized_binned_entropy,
    reference_style_feature_scores,
)
from routing_aware_atos.utils.io import load_json, load_npz, load_yaml, save_json, save_npz


def _iter_layer_batches(
    loader: ActivationLoader,
    *,
    layer: int,
    split_name: str,
    max_tokens: int,
    batch_tokens: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    residual_parts: list[np.ndarray] = []
    token_parts: list[np.ndarray] = []
    buffered = 0
    consumed = 0
    for sample in loader.iter_cached_samples(
        layer_indices=[layer],
        split_name=split_name,
        strict=True,
    ):
        remaining = max_tokens - consumed
        if remaining <= 0:
            break
        residuals = np.asarray(sample.residuals[layer], dtype=np.float32)[:remaining]
        token_ids = np.asarray(sample.tokens, dtype=np.int64)[:remaining]
        residual_parts.append(residuals)
        token_parts.append(token_ids)
        buffered += residuals.shape[0]
        consumed += residuals.shape[0]
        if buffered >= batch_tokens:
            yield np.concatenate(residual_parts), np.concatenate(token_parts)
            residual_parts.clear()
            token_parts.clear()
            buffered = 0
    if residual_parts:
        yield np.concatenate(residual_parts), np.concatenate(token_parts)
    if consumed < max_tokens:
        raise RuntimeError(
            f"Requested {max_tokens} feature-selection tokens from split {split_name!r}, found {consumed}"
        )


class _TorchSAEEncoder:
    def __init__(self, arrays: dict[str, np.ndarray], device: str):
        validate_sae_artifact(arrays)
        if "encoder" not in arrays:
            raise ValueError("Exported SAE artifact is missing encoder weights")
        normalization = (
            str(np.asarray(arrays["normalize_activations"]).item())
            if "normalize_activations" in arrays
            else "none"
        )
        if normalization not in {"none", "None"}:
            raise ValueError(f"Unsupported SAE activation normalization {normalization!r}")
        self.device = torch.device(device)
        self.encoder = torch.as_tensor(arrays["encoder"], dtype=torch.float32, device=self.device)
        apply_b_dec = (
            bool(np.asarray(arrays["apply_b_dec_to_input"]).item())
            if "apply_b_dec_to_input" in arrays
            else True
        )
        self.b_dec = (
            torch.as_tensor(arrays["b_dec"], dtype=torch.float32, device=self.device)
            if "b_dec" in arrays and apply_b_dec
            else None
        )
        self.b_enc = (
            torch.as_tensor(arrays["b_enc"], dtype=torch.float32, device=self.device)
            if "b_enc" in arrays
            else None
        )
        self.threshold = (
            torch.as_tensor(arrays["threshold"], dtype=torch.float32, device=self.device)
            if "threshold" in arrays
            else None
        )

    @property
    def num_features(self) -> int:
        return int(self.encoder.shape[1])

    def parameters_for(
        self,
        feature_ids: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if feature_ids is None:
            return self.encoder, self.b_enc, self.threshold
        ids = torch.as_tensor(feature_ids, dtype=torch.long, device=self.device)
        return (
            self.encoder[:, ids],
            None if self.b_enc is None else self.b_enc[ids],
            None
            if self.threshold is None
            else self.threshold
            if self.threshold.ndim == 0
            else self.threshold[ids],
        )

    def encode(
        self,
        residuals: np.ndarray | torch.Tensor,
        feature_ids: np.ndarray | None = None,
        *,
        parameters: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        x = torch.as_tensor(residuals, dtype=torch.float32, device=self.device)
        if self.b_dec is not None:
            x = x - self.b_dec
        if parameters is not None and feature_ids is not None:
            raise ValueError("Pass feature_ids or prepared parameters, not both")
        encoder, b_enc, threshold = parameters or self.parameters_for(feature_ids)
        pre = x @ encoder
        if b_enc is not None:
            pre = pre + b_enc
        if threshold is not None:
            return torch.where(pre > threshold, torch.relu(pre), torch.zeros_like(pre))
        return torch.relu(pre)


def _sampled_decoder_cosine(
    decoder: np.ndarray,
    feature_ids: np.ndarray,
    *,
    device: str,
    sample_size: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    anchor_count = min(int(sample_size), feature_ids.size)
    anchor_columns = np.sort(rng.choice(feature_ids.size, size=anchor_count, replace=False))
    anchor_ids = feature_ids[anchor_columns]
    anchors = torch.as_tensor(decoder[anchor_ids], dtype=torch.float32, device=device)
    anchors = F.normalize(anchors, dim=1)
    anchor_lookup = {int(feature_id): column for column, feature_id in enumerate(anchor_ids)}
    result = np.zeros(feature_ids.size, dtype=np.float32)
    for start in range(0, feature_ids.size, batch_size):
        stop = min(start + batch_size, feature_ids.size)
        batch_ids = feature_ids[start:stop]
        selected = torch.as_tensor(decoder[batch_ids], dtype=torch.float32, device=device)
        similarities = torch.abs(F.normalize(selected, dim=1) @ anchors.T)
        for row, feature_id in enumerate(batch_ids):
            anchor_column = anchor_lookup.get(int(feature_id))
            if anchor_column is not None:
                similarities[row, anchor_column] = 0.0
        result[start:stop] = similarities.max(dim=1).values.cpu().numpy()
        del selected, similarities
    del anchors
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _resolve_dtype(name: str) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model dtype {name!r}") from exc


def _load_reference_model(cfg: dict, device: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional real-model dependency
        raise ImportError("Install the real-model extra for reference-style feature selection") from exc

    model_name = str(cfg["model_name"])
    revision = cfg.get("model_revision")
    token = cfg.get("token")
    tokenizer_kwargs = {"revision": revision, "token": token}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        **{key: value for key, value in tokenizer_kwargs.items() if value is not None},
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "token": token,
        "dtype": _resolve_dtype(str(cfg.get("model_dtype", "float32"))),
        "attn_implementation": cfg.get("attn_implementation", "eager"),
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **{key: value for key, value in model_kwargs.items() if value is not None},
    )
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def _compute_logit_focus(
    decoder: np.ndarray,
    feature_ids: np.ndarray,
    *,
    output_embedding: torch.Tensor,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    focus = np.zeros(feature_ids.size, dtype=np.float32)
    entropy = np.zeros(feature_ids.size, dtype=np.float32)
    vocabulary_weight = output_embedding.detach().to(device=device, dtype=torch.float32)
    for start in range(0, feature_ids.size, batch_size):
        stop = min(start + batch_size, feature_ids.size)
        vectors = (
            torch.as_tensor(decoder[feature_ids[start:stop]], dtype=torch.float32, device=device)
            @ vocabulary_weight.T
        )
        means = vectors.mean(dim=1)
        standard_deviations = vectors.std(dim=1, unbiased=False).clamp_min(1e-6)
        focus[start:stop] = ((vectors.max(dim=1).values - means) / standard_deviations).cpu().numpy()
        log_probabilities = torch.log_softmax(vectors, dim=1)
        entropy[start:stop] = (
            -(log_probabilities.exp() * log_probabilities).sum(dim=1)
        ).cpu().numpy()
        del vectors, means, standard_deviations, log_probabilities
    return focus, entropy


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise RuntimeError(f"Unsupported transformer-layer output type {type(output)}")


def _hidden_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError(f"Unsupported transformer-layer output type {type(output)}")


def _causal_cross_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    token_losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(input_ids.shape[0], -1)
    valid = attention_mask[:, 1:].to(dtype=torch.bool)
    return token_losses[valid].mean()


@torch.inference_mode()
def _causal_effect_scores(
    model: Any,
    tokenizer: Any,
    *,
    layer: int,
    candidate_ids: np.ndarray,
    decoder: np.ndarray,
    encoder: _TorchSAEEncoder,
    prompts: list[str],
    device: str,
) -> np.ndarray:
    if not prompts:
        raise ValueError("causal_probe_prompts cannot be empty in reference_full mode")
    encoded = tokenizer(
        prompts,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    baseline_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    baseline_loss = _causal_cross_entropy(baseline_logits, input_ids, attention_mask)
    del baseline_logits

    try:
        target_module = model.model.layers[layer]
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(f"Could not resolve post-layer residual module model.layers.{layer}") from exc

    effects = np.zeros(candidate_ids.size, dtype=np.float32)
    decoder_tensor = torch.as_tensor(decoder, dtype=torch.float32, device=device)
    for column, raw_feature_id in enumerate(candidate_ids):
        feature_id = int(raw_feature_id)
        encoder_vector = encoder.encoder[:, feature_id]
        decoder_vector = decoder_tensor[feature_id]
        encoder_bias = None if encoder.b_enc is None else encoder.b_enc[feature_id]
        threshold = (
            None
            if encoder.threshold is None
            else encoder.threshold
            if encoder.threshold.ndim == 0
            else encoder.threshold[feature_id]
        )

        def ablate_feature(module, inputs, output):
            hidden = _hidden_from_output(output)
            centered = hidden if encoder.b_dec is None else hidden - encoder.b_dec
            preactivation = centered @ encoder_vector
            if encoder_bias is not None:
                preactivation = preactivation + encoder_bias
            activation = torch.relu(preactivation)
            if threshold is not None:
                activation = torch.where(
                    preactivation > threshold,
                    activation,
                    torch.zeros_like(activation),
                )
            return _replace_hidden(output, hidden - activation.unsqueeze(-1) * decoder_vector)

        handle = target_module.register_forward_hook(ablate_feature)
        try:
            ablated_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            ablated_loss = _causal_cross_entropy(ablated_logits, input_ids, attention_mask)
            effects[column] = float((ablated_loss - baseline_loss).cpu())
            del ablated_logits, ablated_loss
        finally:
            handle.remove()
        if (column + 1) % 100 == 0 or column + 1 == candidate_ids.size:
            print(f"Layer {layer}: causal feature {column + 1}/{candidate_ids.size}", flush=True)
    del decoder_tensor
    return effects


def _resolve_layer_configs(cfg: dict) -> tuple[str, list[int], dict[int, str], dict]:
    selection_cfg = dict(cfg.get("feature_selection", cfg))
    for key, fallback in (
        ("model_name", cfg.get("model_name")),
        ("model_revision", cfg.get("model_revision")),
        ("model_dtype", cfg.get("dtype", "float32")),
        ("attn_implementation", cfg.get("attn_implementation", "eager")),
    ):
        selection_cfg.setdefault(key, fallback)
    activation_dir = str(
        selection_cfg.get("activation_dir_path")
        or cfg.get("collection", {}).get("output_dir")
    )
    raw_layers = selection_cfg.get("target_layers")
    if raw_layers is None:
        raw_layer = selection_cfg.get("target_layer")
        raw_layers = [] if raw_layer is None else [raw_layer]
    if not raw_layers:
        raise ValueError("feature_selection.target_layers is required")
    layers = [int(layer) for layer in raw_layers]
    artifact_map = selection_cfg.get("sae_artifacts") or cfg.get("sae", {}).get("artifacts", {})
    artifacts = {int(layer): str(path) for layer, path in artifact_map.items()}
    missing = [layer for layer in layers if layer not in artifacts]
    if missing:
        raise ValueError(f"Missing SAE artifact paths for target layers {missing}")
    return activation_dir, layers, artifacts, selection_cfg


def _selection_provenance(cfg: dict, artifact_path: str) -> dict[str, Any]:
    manifest_path = Path(cfg["activation_dir_path"]) / "manifest.json"
    activation_manifest = load_json(manifest_path) if manifest_path.exists() else {}
    return {
        "sae_artifact_sha256": sha256_file(artifact_path),
        "model_name": cfg.get("model_name"),
        "model_revision": cfg.get("model_revision"),
        "activation_config_sha256": activation_manifest.get("config_sha256"),
        "selection_config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "implementation_sha256": feature_selection_implementation_sha256(),
    }


def _select_for_layer(
    loader: ActivationLoader,
    *,
    layer: int,
    artifact_path: str,
    cfg: dict,
    reference_model: Any | None = None,
    tokenizer: Any | None = None,
) -> dict:
    arrays = load_npz(artifact_path)
    decoder = np.asarray(arrays["decoder"], dtype=np.float32)
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    encoder = _TorchSAEEncoder(arrays, device)
    max_tokens = int(cfg.get("max_tokens", 120_000))
    batch_tokens = int(cfg.get("batch_tokens", 512))
    split_name = str(cfg.get("split_name", "train"))
    num_features = encoder.num_features

    counts = np.zeros(num_features, dtype=np.int64)
    sums = np.zeros(num_features, dtype=np.float64)
    sum_squares = np.zeros(num_features, dtype=np.float64)
    maxima = np.zeros(num_features, dtype=np.float32)
    processed = 0
    for residuals, _ in _iter_layer_batches(
        loader,
        layer=layer,
        split_name=split_name,
        max_tokens=max_tokens,
        batch_tokens=batch_tokens,
    ):
        activations = encoder.encode(residuals)
        active = activations > 0
        counts += active.sum(dim=0).cpu().numpy().astype(np.int64)
        sums += activations.sum(dim=0).cpu().numpy().astype(np.float64)
        sum_squares += (activations * activations).sum(dim=0).cpu().numpy().astype(np.float64)
        maxima = np.maximum(maxima, activations.max(dim=0).values.cpu().numpy())
        processed += residuals.shape[0]
        del activations, active

    minimum_count = int(cfg.get("min_activation_count", 10))
    max_firing_rate = float(cfg.get("max_firing_rate", 0.20))
    eligible = np.flatnonzero((counts >= minimum_count) & (counts <= processed * max_firing_rate))
    if eligible.size == 0:
        raise RuntimeError(f"No eligible SAE features at layer {layer}")
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.maximum(
        np.divide(sum_squares, counts, out=np.zeros_like(sum_squares), where=counts > 0) - means**2,
        0.0,
    )
    activity_strength = np.log1p(counts) * np.sqrt(np.maximum(means, 0.0) * np.sqrt(variances + 1e-12))

    hash_bins = int(cfg.get("token_hash_bins", 2048))
    token_counts = np.zeros((eligible.size, hash_bins), dtype=np.uint32)
    prepared = encoder.parameters_for(eligible)
    redundancy_mode = str(cfg.get("redundancy_mode", "activation_correlation"))
    redundancy_accumulator = None
    redundancy_anchor_columns = np.empty(0, dtype=np.int64)
    eligible_global_mean = None
    if redundancy_mode == "activation_correlation":
        redundancy_sample_size = min(int(cfg.get("redundancy_sample_size", 512)), eligible.size)
        rng = np.random.default_rng(int(cfg.get("selection_seed", 0)))
        redundancy_anchor_columns = np.sort(
            rng.choice(eligible.size, size=redundancy_sample_size, replace=False)
        )
        redundancy_accumulator = torch.zeros(
            (redundancy_sample_size, eligible.size),
            dtype=torch.float32,
            device=encoder.device,
        )
        eligible_global_mean = torch.as_tensor(
            sums[eligible] / float(processed),
            dtype=torch.float32,
            device=encoder.device,
        )
    elif redundancy_mode != "decoder_cosine":
        raise ValueError(f"Unsupported feature-selection redundancy_mode {redundancy_mode!r}")

    for residuals, token_ids in _iter_layer_batches(
        loader,
        layer=layer,
        split_name=split_name,
        max_tokens=max_tokens,
        batch_tokens=batch_tokens,
    ):
        activations = encoder.encode(residuals, parameters=prepared)
        row_ids, feature_columns = torch.nonzero(activations > 0, as_tuple=True)
        if row_ids.numel():
            rows_np = row_ids.cpu().numpy()
            columns_np = feature_columns.cpu().numpy()
            buckets = np.mod(token_ids[rows_np], hash_bins)
            np.add.at(token_counts, (columns_np, buckets), 1)
        if redundancy_accumulator is not None:
            assert eligible_global_mean is not None
            centered = activations - eligible_global_mean
            anchor_tensor = torch.as_tensor(
                redundancy_anchor_columns,
                dtype=torch.long,
                device=encoder.device,
            )
            redundancy_accumulator += centered[:, anchor_tensor].T @ centered
            del centered, anchor_tensor
        del activations, row_ids, feature_columns

    method = str(cfg.get("method", "fast_proxy"))
    entropy = (
        binned_entropy(token_counts)
        if method == "reference_full"
        else normalized_binned_entropy(token_counts)
    )
    if redundancy_accumulator is not None:
        centered_sum_squares = np.maximum(
            sum_squares[eligible] - sums[eligible] ** 2 / float(processed),
            0.0,
        )
        redundancy = max_abs_anchor_correlation(
            redundancy_accumulator.cpu().numpy(),
            centered_sum_squares,
            redundancy_anchor_columns,
        )
        del redundancy_accumulator
        del eligible_global_mean
    else:
        redundancy = _sampled_decoder_cosine(
            decoder,
            eligible,
            device=device,
            sample_size=int(cfg.get("redundancy_sample_size", 512)),
            seed=int(cfg.get("selection_seed", 0)),
            batch_size=int(cfg.get("redundancy_batch_size", 1024)),
        )

    vocabulary_focus = np.zeros(eligible.size, dtype=np.float32)
    vocabulary_entropy = np.zeros(eligible.size, dtype=np.float32)
    causal_effect = np.zeros(eligible.size, dtype=np.float32)
    candidate_mask = np.zeros(eligible.size, dtype=bool)
    if method == "reference_full":
        if reference_model is None or tokenizer is None:
            raise RuntimeError("reference_full feature selection requires the language model and tokenizer")
        output_embeddings = reference_model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise RuntimeError("The language model does not expose output embedding weights")
        vocabulary_focus, vocabulary_entropy = _compute_logit_focus(
            decoder,
            eligible,
            output_embedding=output_embeddings.weight,
            device=device,
            batch_size=int(cfg.get("logit_focus_batch_size", 32)),
        )
        pre_scores = reference_style_feature_scores(
            token_entropy=entropy,
            vocabulary_focus=vocabulary_focus,
            redundancy=redundancy,
            activation_rate=counts[eligible] / processed,
        )
        candidate_percent = float(cfg.get("candidate_percent", 15.0))
        candidate_count = min(
            eligible.size,
            max(32, int(num_features * candidate_percent / 100.0)),
        )
        candidate_columns = np.argsort(pre_scores)[-candidate_count:][::-1]
        candidate_ids = eligible[candidate_columns]
        candidate_mask[candidate_columns] = True
        causal_effect[candidate_columns] = _causal_effect_scores(
            reference_model,
            tokenizer,
            layer=layer,
            candidate_ids=candidate_ids,
            decoder=decoder,
            encoder=encoder,
            prompts=[str(prompt) for prompt in cfg.get("causal_probe_prompts", [])],
            device=device,
        )
        scores = reference_style_feature_scores(
            token_entropy=entropy,
            vocabulary_focus=vocabulary_focus,
            redundancy=redundancy,
            activation_rate=counts[eligible] / processed,
            causal_effect=causal_effect,
            causal_candidate_mask=candidate_mask,
        )
        method_name = "reference_style_with_live_causal_ablation"
    elif method == "fast_proxy":
        candidate_ids = eligible
        candidate_count = eligible.size
        weights = cfg.get("score_weights", {})
        scores = composite_feature_scores(
            activation_strength=activity_strength[eligible],
            token_entropy=entropy,
            max_decoder_cosine=redundancy,
            coherence_weight=float(weights.get("coherence", 0.45)),
            uniqueness_weight=float(weights.get("uniqueness", 0.25)),
            activity_weight=float(weights.get("activity", 0.30)),
        )
        method_name = "activity_token_entropy_decoder_uniqueness"
    else:
        raise ValueError(f"Unsupported feature-selection method {method!r}")

    top_percent = float(cfg.get("top_percent", 5.0))
    selected_count = min(eligible.size, max(1, int(num_features * top_percent / 100.0)))
    selected_columns = np.argsort(scores)[-selected_count:][::-1]
    selected_ids = eligible[selected_columns]

    output_dir = Path(cfg.get("output_dir", "artifacts/features"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"layer_{layer}_features.json"
    output_stats = output_dir / f"layer_{layer}_feature_stats.npz"
    provenance = _selection_provenance(cfg, artifact_path)
    records = []
    for column in selected_columns:
        feature_id = int(eligible[column])
        records.append(
            {
                "feature_id": feature_id,
                "score": float(scores[column]),
                "activation_count": int(counts[feature_id]),
                "firing_rate": float(counts[feature_id] / processed),
                "mean_activation": float(means[feature_id]),
                "max_activation": float(maxima[feature_id]),
                "token_entropy": float(entropy[column]),
                "redundancy": float(redundancy[column]),
                "vocabulary_focus": float(vocabulary_focus[column]),
                "vocabulary_entropy": float(vocabulary_entropy[column]),
                "causal_effect_delta_loss": float(causal_effect[column]),
                "causal_candidate": bool(candidate_mask[column]),
            }
        )
    payload = {
        "layer": layer,
        "sae_artifact": artifact_path,
        **provenance,
        "split": split_name,
        "tokens_processed": processed,
        "num_sae_features": num_features,
        "eligible_count": int(eligible.size),
        "candidate_count": int(candidate_count),
        "selected_count": int(selected_ids.size),
        "high_quality_feature_ids": selected_ids.astype(int).tolist(),
        "features": records,
        "method": method_name,
        "redundancy_mode": redundancy_mode,
        "causal_probe_prompts": (
            [str(prompt) for prompt in cfg.get("causal_probe_prompts", [])]
            if method == "reference_full"
            else []
        ),
        "implementation_notes": (
            "The full mode follows the released selector's score weights and live feature-ablation test. "
            "Token coherence uses deterministic hashed token bins to bound memory."
            if method == "reference_full"
            else "Fast proxy mode omits model-based vocabulary focus and causal ablation."
        ),
    }
    save_json(output_json, payload)
    save_npz(
        output_stats,
        activation_count=counts,
        activation_sum=sums,
        activation_sum_squares=sum_squares,
        max_activation=maxima,
        eligible_ids=eligible,
        candidate_ids=candidate_ids,
        eligible_token_entropy=entropy,
        eligible_redundancy=redundancy,
        eligible_vocabulary_focus=vocabulary_focus,
        eligible_vocabulary_entropy=vocabulary_entropy,
        eligible_causal_effect=causal_effect,
        eligible_scores=scores,
        selected_ids=selected_ids,
    )
    print(f"Selected {selected_ids.size} SAE features for layer {layer} -> {output_json}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Select reproducible high-quality Gemma Scope features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    activation_dir, layers, artifacts, selection_cfg = _resolve_layer_configs(cfg)
    loader = ActivationLoader(activation_dir_path=activation_dir)
    reference_model = None
    tokenizer = None
    try:
        expected_method = {
            "reference_full": "reference_style_with_live_causal_ablation",
            "fast_proxy": "activity_token_entropy_decoder_uniqueness",
        }[str(selection_cfg.get("method", "fast_proxy"))]
        output_dir = Path(selection_cfg.get("output_dir", "artifacts/features"))
        pending_layers: list[int] = []
        for layer in layers:
            output_path = output_dir / f"layer_{layer}_features.json"
            stats_path = output_dir / f"layer_{layer}_feature_stats.npz"
            try:
                current = load_json(output_path) if output_path.exists() else {}
            except (OSError, ValueError):
                current = {}
            provenance = _selection_provenance(selection_cfg, artifacts[layer])
            try:
                stats = load_npz(stats_path) if stats_path.exists() else {}
            except (OSError, ValueError):
                stats = {}
            stats_match = bool(
                "selected_ids" in stats
                and stats["selected_ids"].astype(int).tolist()
                == current.get("high_quality_feature_ids")
            )
            if (
                not args.force
                and stats_match
                and current.get("method") == expected_method
                and all(current.get(key) == value for key, value in provenance.items())
            ):
                print(f"SKIP: current feature artifact already exists -> {output_path}")
            else:
                pending_layers.append(layer)
        if pending_layers and str(selection_cfg.get("method", "fast_proxy")) == "reference_full":
            device = str(
                selection_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
            reference_model, tokenizer = _load_reference_model(selection_cfg, device)
        for layer in pending_layers:
            _select_for_layer(
                loader,
                layer=layer,
                artifact_path=artifacts[layer],
                cfg=selection_cfg,
                reference_model=reference_model,
                tokenizer=tokenizer,
            )
    finally:
        loader.close()
        del reference_model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
