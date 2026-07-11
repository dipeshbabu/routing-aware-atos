from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from importlib.metadata import version
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch

from routing_aware_atos.data.activation_store import (
    SPLIT_IDS,
    ActivationShardWriter,
    ActivationStoreSpec,
)
from routing_aware_atos.data.slimpajama import (
    PackedSequence,
    collect_document_token_budget,
    collect_full_document_sequences,
    iter_document_token_ids,
    pack_token_budget,
    split_counts,
    split_token_budgets,
)
from routing_aware_atos.provenance import collection_implementation_sha256
from routing_aware_atos.routed_types import CachedSample
from routing_aware_atos.utils.io import load_yaml, save_cached_samples, save_json


def _load_prompts(path: str | Path) -> list[str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(x) for x in payload]
        if isinstance(payload, dict) and "prompts" in payload:
            return [str(x) for x in payload["prompts"]]
        raise ValueError("JSON prompts file must be a list or contain a 'prompts' list")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return dtype


def _dataset_stream(
    dataset_cfg: dict[str, Any],
    *,
    split: str,
    seed: int,
    skip_examples: int | None = None,
    shuffle_buffer_size: int | None = None,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional real-model dependency
        raise ImportError("Install the real-model extra to stream SlimPajama") from exc

    kwargs: dict[str, Any] = {
        "path": dataset_cfg["name"],
        "split": split,
        "streaming": True,
    }
    if dataset_cfg.get("config_name"):
        kwargs["name"] = dataset_cfg["config_name"]
    if dataset_cfg.get("revision"):
        kwargs["revision"] = dataset_cfg["revision"]
    if dataset_cfg.get("data_files"):
        kwargs["data_files"] = dataset_cfg["data_files"]

    dataset = load_dataset(**kwargs)
    skip_examples = int(
        dataset_cfg.get("skip_examples", 0)
        if skip_examples is None
        else skip_examples
    )
    if skip_examples:
        dataset = dataset.skip(skip_examples)
    buffer_size = int(
        dataset_cfg.get("shuffle_buffer_size", 10_000)
        if shuffle_buffer_size is None
        else shuffle_buffer_size
    )
    if buffer_size > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    return dataset


def _build_dataset_sequences(cfg: dict[str, Any], tokenizer: Any) -> list[PackedSequence]:
    dataset_cfg = cfg["dataset"]
    collection_cfg = cfg.get("collection", cfg)
    sequence_length = int(collection_cfg.get("sequence_length", collection_cfg.get("max_length", 256)))
    total_tokens = int(collection_cfg.get("operator_tokens", 250_000))
    fractions = collection_cfg.get(
        "split_fractions",
        {"train": 0.6, "validation": 0.2, "test": 0.2},
    )
    budgets = split_token_budgets(total_tokens, fractions)
    seed = int(dataset_cfg.get("seed", 42))
    source_split = str(dataset_cfg.get("operator_source_split", "train"))
    rows = _dataset_stream(dataset_cfg, split=source_split, seed=seed)
    packing_mode = str(collection_cfg.get("packing_mode", "documents"))
    max_document_tokens = (
        sequence_length - 1
        if packing_mode == "documents"
        else int(collection_cfg.get("max_document_tokens", sequence_length * 16))
    )
    documents = iter_document_token_ids(
        rows,
        tokenizer,
        text_field=str(dataset_cfg.get("text_field", "text")),
        max_tokens=max_document_tokens,
    )

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        raise ValueError("The tokenizer must define bos_token_id")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id or eos_token_id")

    sequences: list[PackedSequence] = []
    if packing_mode not in {"documents", "packed"}:
        raise ValueError("collection.packing_mode must be 'documents' or 'packed'")
    collector = collect_document_token_budget if packing_mode == "documents" else pack_token_budget
    for split_name in ("train", "validation", "test"):
        split_sequences = collector(
            iter(documents),
            token_budget=budgets[split_name],
            sequence_length=sequence_length,
            bos_token_id=int(bos_token_id),
            pad_token_id=int(pad_token_id),
            split=split_name,
            source_split=source_split,
            start_sequence_index=len(sequences),
        )
        sequences.extend(split_sequences)

    causal_sequences = int(collection_cfg.get("causal_sequences", 100))
    if causal_sequences > 0:
        causal_source_split = str(dataset_cfg.get("causal_source_split", "test"))
        causal_rows = _dataset_stream(
            dataset_cfg,
            split=causal_source_split,
            seed=seed + int(dataset_cfg.get("causal_seed_offset", 1)),
            skip_examples=int(dataset_cfg.get("causal_skip_examples", 10_000)),
            shuffle_buffer_size=int(dataset_cfg.get("causal_shuffle_buffer_size", 1)),
        )
        causal_documents = iter_document_token_ids(
            causal_rows,
            tokenizer,
            text_field=str(dataset_cfg.get("text_field", "text")),
            max_tokens=max_document_tokens,
        )
        if packing_mode == "documents":
            causal = collect_full_document_sequences(
                iter(causal_documents),
                num_sequences=causal_sequences,
                sequence_length=sequence_length,
                bos_token_id=int(bos_token_id),
                split="causal",
                source_split=causal_source_split,
                start_sequence_index=len(sequences),
            )
        else:
            causal = pack_token_budget(
                iter(causal_documents),
                token_budget=causal_sequences * sequence_length,
                sequence_length=sequence_length,
                bos_token_id=int(bos_token_id),
                pad_token_id=int(pad_token_id),
                split="causal",
                source_split=causal_source_split,
                start_sequence_index=len(sequences),
            )
        sequences.extend(causal)
    return sequences


def _load_model_and_tokenizer(cfg: dict[str, Any]):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional real-model dependency
        raise ImportError("Install the real-model extra to use this collector") from exc

    model_name = str(cfg["model_name"])
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _resolve_dtype(str(cfg.get("dtype", "bfloat16")))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=cfg.get("model_revision"),
        token=cfg.get("token"),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "revision": cfg.get("model_revision"),
        "token": cfg.get("token"),
    }
    if cfg.get("attn_implementation") is not None:
        model_kwargs["attn_implementation"] = cfg["attn_implementation"]
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _validate_layers(model: Any, layer_indices: Iterable[int], attention_layers: Iterable[int]) -> int:
    num_layers = int(model.config.num_hidden_layers)
    requested = set(int(layer) for layer in layer_indices) | set(int(layer) for layer in attention_layers)
    invalid = sorted(layer for layer in requested if layer < 0 or layer >= num_layers)
    if invalid:
        raise ValueError(f"Layer indices {invalid} are outside model layers [0, {num_layers - 1}]")
    return num_layers


def _forward_batch(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: list[int],
    attention_layers: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=bool(attention_layers),
            use_cache=False,
            return_dict=True,
        )

    if outputs.hidden_states is None:
        raise RuntimeError("Model did not return hidden states")
    valid_token_mask = attention_mask.unsqueeze(-1)
    residuals = {
        layer_idx: (
            outputs.hidden_states[layer_idx + 1] * valid_token_mask
        ).detach().cpu().numpy()
        for layer_idx in layer_indices
    }

    attention_scores: dict[int, np.ndarray] = {}
    if attention_layers:
        if outputs.attentions is None:
            raise RuntimeError(
                "Model did not return attention tensors. Use attn_implementation: eager."
            )
        for layer_idx in attention_layers:
            pooled = outputs.attentions[layer_idx].mean(dim=1)
            attention_scores[layer_idx] = pooled.detach().cpu().numpy()
    return residuals, attention_scores


def _config_sha256(cfg: dict[str, Any]) -> str:
    collection_protocol = {
        key: cfg.get(key)
        for key in (
            "model_name",
            "model_revision",
            "device",
            "dtype",
            "attn_implementation",
            "dataset",
            "collection",
        )
    }
    collection_protocol["implementation_sha256"] = collection_implementation_sha256()
    return hashlib.sha256(
        json.dumps(collection_protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_shards_match(output_dir: Path, manifest: dict[str, Any]) -> bool:
    records = manifest.get("shards")
    if not isinstance(records, list) or not records:
        return False
    expected_names = {str(record.get("path")) for record in records}
    actual_names = {path.name for path in output_dir.glob("part-*.zip")}
    if expected_names != actual_names:
        return False
    for record in records:
        path = output_dir / str(record.get("path"))
        if not path.is_file() or record.get("sha256") != _file_sha256(path):
            return False
    return True


def _completed_shard_matches(
    path: Path,
    spec: ActivationStoreSpec,
    sequences: list[PackedSequence],
) -> bool:
    try:
        import zarr

        store = zarr.storage.ZipStore(path, mode="r")
        try:
            root = zarr.open_group(store=store, mode="r")
            if "complete" not in root or int(np.asarray(root["complete"][0])) != 1:
                return False
            if int(root.attrs.get("schema_version", -1)) != 3:
                return False
            if root.attrs.get("collection_sha256") != spec.collection_sha256:
                return False
            expected_shapes = {
                "input_ids": (spec.num_samples, spec.sequence_length),
                "attention_mask": (spec.num_samples, spec.sequence_length),
                "split_id": (spec.num_samples,),
                "sequence_index": (spec.num_samples,),
            }
            if any(name not in root or tuple(root[name].shape) != shape for name, shape in expected_shapes.items()):
                return False
            if "activations" not in root or any(
                f"layer_{layer}" not in root["activations"]
                or tuple(root["activations"][f"layer_{layer}"].shape)
                != (spec.num_samples, spec.sequence_length, spec.d_model)
                for layer in spec.layer_indices
            ):
                return False
            if spec.attention_layers and (
                "attention_scores" not in root
                or any(
                    f"attention_layer_{layer}" not in root["attention_scores"]
                    or tuple(root["attention_scores"][f"attention_layer_{layer}"].shape)
                    != (spec.num_samples, spec.sequence_length, spec.sequence_length)
                    for layer in spec.attention_layers
                )
            ):
                return False
            expected_input_ids = np.stack([sequence.input_ids for sequence in sequences])
            expected_masks = np.stack([sequence.attention_mask for sequence in sequences])
            expected_splits = np.asarray([SPLIT_IDS[sequence.split] for sequence in sequences])
            expected_indices = np.asarray([sequence.sequence_index for sequence in sequences])
            return bool(
                np.array_equal(np.asarray(root["input_ids"]), expected_input_ids)
                and np.array_equal(np.asarray(root["attention_mask"]), expected_masks)
                and np.array_equal(np.asarray(root["split_id"]), expected_splits)
                and np.array_equal(np.asarray(root["sequence_index"]), expected_indices)
            )
        finally:
            store.close()
    except Exception:
        return False


def _collect_dataset_cache(
    cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: str,
) -> None:
    collection_cfg = cfg.get("collection", cfg)
    output_dir = Path(collection_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = _build_dataset_sequences(cfg, tokenizer)
    layer_indices = [int(layer) for layer in collection_cfg["layer_indices"]]
    attention_layers = sorted(
        set(
            int(layer)
            for layer in collection_cfg.get(
                "attention_layers",
                [pair[1] for pair in collection_cfg.get("attention_layer_pairs", [])],
            )
        )
    )
    num_layers = _validate_layers(model, layer_indices, attention_layers)
    d_model = int(model.config.hidden_size)
    sequence_length = int(collection_cfg.get("sequence_length", 256))
    batch_size = int(collection_cfg.get("batch_size", 2))
    shard_size = int(collection_cfg.get("shard_size", 8))
    cache_dtype = str(collection_cfg.get("cache_dtype", "float16"))
    if batch_size <= 0 or shard_size <= 0:
        raise ValueError("batch_size and shard_size must be positive")

    config_hash = _config_sha256(cfg)
    state_path = output_dir / "collection_state.json"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing_manifest.get("config_sha256") == config_hash
            and _manifest_shards_match(output_dir, existing_manifest)
        ):
            print(f"Activation cache is already complete -> {manifest_path}")
            return
        if existing_manifest.get("config_sha256") == config_hash:
            raise RuntimeError(
                f"Activation cache at {output_dir} has missing or corrupted shards; "
                "move it aside and recollect into a clean output directory"
            )
        raise FileExistsError(
            f"Activation cache at {output_dir} was produced by a different configuration"
        )

    state = {
        "schema_version": 1,
        "config_sha256": config_hash,
        "implementation_sha256": collection_implementation_sha256(),
        "num_samples": len(sequences),
        "sequence_length": sequence_length,
        "d_model": d_model,
        "layer_indices": layer_indices,
        "attention_layers": attention_layers,
        "cache_dtype": cache_dtype,
        "shard_size": shard_size,
        "split_counts": split_counts(sequences),
    }
    existing_parts = sorted(output_dir.glob("part-*.zip"))
    for partial_path in output_dir.glob(".part-*.partial"):
        partial_path.unlink()
    unexpected = [
        path.name
        for path in output_dir.iterdir()
        if path.name != state_path.name and path not in existing_parts
    ]
    if unexpected:
        raise FileExistsError(
            f"Activation output directory contains unrelated files: {unexpected[:5]}"
        )
    if existing_parts and not state_path.exists():
        raise FileExistsError(
            f"Found activation shards without {state_path}; refusing to mix unverified artifacts"
        )
    if state_path.exists():
        existing_state = json.loads(state_path.read_text(encoding="utf-8"))
        if existing_state != state:
            raise FileExistsError(
                f"Partial activation cache at {output_dir} belongs to a different configuration"
            )
    else:
        save_json(state_path, state)

    expected_shards = {
        f"part-{shard_id:05d}.zip"
        for shard_id, _ in enumerate(range(0, len(sequences), shard_size))
    }
    unexpected_shards = [path.name for path in existing_parts if path.name not in expected_shards]
    if unexpected_shards:
        raise FileExistsError(f"Unexpected activation shards in {output_dir}: {unexpected_shards[:5]}")

    for shard_id, shard_start in enumerate(range(0, len(sequences), shard_size)):
        shard_sequences = sequences[shard_start : shard_start + shard_size]
        shard_path = output_dir / f"part-{shard_id:05d}.zip"
        spec = ActivationStoreSpec(
            num_samples=len(shard_sequences),
            sequence_length=sequence_length,
            d_model=d_model,
            layer_indices=tuple(layer_indices),
            attention_layers=tuple(attention_layers),
            cache_dtype=cache_dtype,
            collection_sha256=config_hash,
        )
        if shard_path.exists() and _completed_shard_matches(shard_path, spec, shard_sequences):
            print(
                f"SKIP: verified activation shard {shard_id + 1}/"
                f"{math.ceil(len(sequences) / shard_size)} -> {shard_path}"
            )
            continue
        temporary_shard_path = output_dir / f".{shard_path.stem}.partial"
        try:
            with ActivationShardWriter(temporary_shard_path, spec) as writer:
                for local_start in range(0, len(shard_sequences), batch_size):
                    batch = shard_sequences[local_start : local_start + batch_size]
                    input_ids_np = np.stack([sequence.input_ids for sequence in batch])
                    attention_mask_np = np.stack([sequence.attention_mask for sequence in batch])
                    input_ids = torch.from_numpy(input_ids_np.astype(np.int64)).to(device)
                    attention_mask = torch.from_numpy(attention_mask_np.astype(np.int64)).to(device)
                    residuals, attention_scores = _forward_batch(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        layer_indices=layer_indices,
                        attention_layers=attention_layers,
                    )
                    writer.write_batch(
                        local_start,
                        input_ids=input_ids_np,
                        attention_mask=attention_mask_np,
                        split_ids=np.asarray([SPLIT_IDS[sequence.split] for sequence in batch]),
                        sequence_indices=np.asarray([sequence.sequence_index for sequence in batch]),
                        residuals=residuals,
                        attention_scores=attention_scores,
                    )
                    del input_ids, attention_mask, residuals, attention_scores
            os.replace(temporary_shard_path, shard_path)
        finally:
            temporary_shard_path.unlink(missing_ok=True)
        print(f"Saved activation shard {shard_id + 1}/{math.ceil(len(sequences) / shard_size)} -> {shard_path}")

    dataset_cfg = cfg["dataset"]
    shard_records = []
    for shard_id, shard_start in enumerate(range(0, len(sequences), shard_size)):
        shard_path = output_dir / f"part-{shard_id:05d}.zip"
        shard_records.append(
            {
                "path": shard_path.name,
                "sha256": _file_sha256(shard_path),
                "num_samples": min(shard_size, len(sequences) - shard_start),
            }
        )
    manifest = {
        "schema_version": 3,
        "model_name": cfg["model_name"],
        "model_revision": cfg.get("model_revision"),
        "dataset": {
            "name": dataset_cfg["name"],
            "config_name": dataset_cfg.get("config_name"),
            "revision": dataset_cfg.get("revision"),
            "operator_source_split": dataset_cfg.get("operator_source_split", "train"),
            "causal_source_split": dataset_cfg.get("causal_source_split", "test"),
            "seed": int(dataset_cfg.get("seed", 42)),
            "causal_seed_offset": int(dataset_cfg.get("causal_seed_offset", 1)),
            "shuffle_buffer_size": int(dataset_cfg.get("shuffle_buffer_size", 10_000)),
            "causal_shuffle_buffer_size": int(
                dataset_cfg.get("causal_shuffle_buffer_size", 1)
            ),
            "skip_examples": int(dataset_cfg.get("skip_examples", 0)),
            "causal_skip_examples": int(dataset_cfg.get("causal_skip_examples", 10_000)),
        },
        "layer_semantics": "post_layer",
        "hidden_state_offset": 1,
        "num_model_layers": num_layers,
        "d_model": d_model,
        "sequence_length": sequence_length,
        "packing_mode": str(collection_cfg.get("packing_mode", "documents")),
        "layer_indices": layer_indices,
        "attention_layers": attention_layers,
        "cache_dtype": cache_dtype,
        "num_samples": len(sequences),
        "shards": shard_records,
        "split_counts": split_counts(sequences),
        "operator_tokens": int(collection_cfg.get("operator_tokens", 250_000)),
        "split_fractions": collection_cfg.get(
            "split_fractions",
            {"train": 0.6, "validation": 0.2, "test": 0.2},
        ),
        "causal_sequences": int(collection_cfg.get("causal_sequences", 100)),
        "config_sha256": config_hash,
        "software_versions": {
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "datasets": version("datasets"),
            "zarr": version("zarr"),
        },
    }
    save_json(output_dir / "manifest.json", manifest)
    print(f"Saved activation manifest -> {output_dir / 'manifest.json'}")


def _collect_prompt_cache(
    cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: str,
) -> None:
    """Legacy small-prompt collector retained for local smoke experiments."""

    layer_indices = [int(layer) for layer in cfg["layer_indices"]]
    attention_pairs = [tuple(int(value) for value in pair) for pair in cfg.get("attention_layer_pairs", [])]
    attention_layers = sorted({target for _, target in attention_pairs})
    _validate_layers(model, layer_indices, attention_layers)
    prompts = _load_prompts(cfg["prompts_path"])
    max_length = int(cfg.get("max_length", 256))
    samples: list[CachedSample] = []

    for sample_idx, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
        residuals, attention_by_layer = _forward_batch(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_indices=layer_indices,
            attention_layers=attention_layers,
        )
        attention_scores = {
            pair: attention_by_layer[pair[1]][0].astype(np.float32)
            for pair in attention_pairs
        }
        samples.append(
            CachedSample(
                tokens=input_ids[0].detach().cpu().tolist(),
                residuals={key: value[0].astype(np.float32) for key, value in residuals.items()},
                attention_scores=attention_scores or None,
                attribution_scores=None,
                metadata={
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "model_name": cfg["model_name"],
                    "split": "train",
                    "layer_semantics": "post_layer",
                },
            )
        )
    save_cached_samples(cfg["output_path"], samples)
    print(f"Saved {len(samples)} cached samples -> {cfg['output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect post-layer residual and pooled-attention caches from a Hugging Face causal LM"
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    model, tokenizer, device = _load_model_and_tokenizer(cfg)
    if cfg.get("dataset"):
        _collect_dataset_cache(cfg, model, tokenizer, device)
    else:
        _collect_prompt_cache(cfg, model, tokenizer, device)


if __name__ == "__main__":
    main()
