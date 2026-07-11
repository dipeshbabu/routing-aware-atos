from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import numpy as np


SPLIT_IDS = {
    "train": 0,
    "validation": 1,
    "test": 2,
    "causal": 3,
}
SPLIT_NAMES = {value: key for key, value in SPLIT_IDS.items()}


def _require_zarr():
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:  # pragma: no cover - core dependency in normal installs
        raise ImportError("zarr and numcodecs are required for activation storage") from exc
    return zarr, Blosc


@dataclass(frozen=True)
class ActivationStoreSpec:
    num_samples: int
    sequence_length: int
    d_model: int
    layer_indices: tuple[int, ...]
    attention_layers: tuple[int, ...] = ()
    cache_dtype: str = "float16"
    collection_sha256: str | None = None

    def validate(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.sequence_length <= 0 or self.d_model <= 0:
            raise ValueError("sequence_length and d_model must be positive")
        if not self.layer_indices:
            raise ValueError("layer_indices cannot be empty")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be float16 or float32")
        if self.collection_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.collection_sha256
        ):
            raise ValueError("collection_sha256 must be a lowercase SHA-256 digest")


class ActivationShardWriter:
    """Incrementally write one compressed Zarr zip shard."""

    def __init__(self, path: str | Path, spec: ActivationStoreSpec):
        spec.validate()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self._written = np.zeros(spec.num_samples, dtype=bool)

        zarr, Blosc = _require_zarr()
        self._store = zarr.storage.ZipStore(self.path, mode="w", compression=0)
        self._root = zarr.open_group(
            store=self._store,
            mode="a",
            zarr_format=2,
            attributes={
                "schema_version": 3,
                "split_names": {str(key): value for key, value in SPLIT_NAMES.items()},
                "layer_semantics": "post_layer",
                "layer_indices": list(spec.layer_indices),
                "attention_layers": list(spec.attention_layers),
                "cache_dtype": spec.cache_dtype,
                "collection_sha256": spec.collection_sha256,
            },
        )
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        sample_chunks = (1, spec.sequence_length)
        activation_chunks = (1, spec.sequence_length, spec.d_model)
        score_chunks = (1, spec.sequence_length, spec.sequence_length)

        self._root.create_array(
            "input_ids",
            shape=(spec.num_samples, spec.sequence_length),
            dtype="i4",
            chunks=sample_chunks,
            compressor=compressor,
        )
        self._root.create_array(
            "attention_mask",
            shape=(spec.num_samples, spec.sequence_length),
            dtype="u1",
            chunks=sample_chunks,
            compressor=compressor,
        )
        self._root.create_array(
            "split_id",
            shape=(spec.num_samples,),
            dtype="u1",
            chunks=(min(256, spec.num_samples),),
            compressor=compressor,
        )
        self._root.create_array(
            "sequence_index",
            shape=(spec.num_samples,),
            dtype="i8",
            chunks=(min(256, spec.num_samples),),
            compressor=compressor,
        )
        self._root.create_array(
            "complete",
            shape=(1,),
            dtype="u1",
            chunks=(1,),
            fill_value=0,
            compressor=compressor,
        )

        activations = self._root.create_group("activations")
        for layer_idx in spec.layer_indices:
            activations.create_array(
                f"layer_{layer_idx}",
                shape=(spec.num_samples, spec.sequence_length, spec.d_model),
                dtype=spec.cache_dtype,
                chunks=activation_chunks,
                compressor=compressor,
            )

        if spec.attention_layers:
            scores = self._root.create_group("attention_scores")
            for layer_idx in spec.attention_layers:
                scores.create_array(
                    f"attention_layer_{layer_idx}",
                    shape=(spec.num_samples, spec.sequence_length, spec.sequence_length),
                    dtype=spec.cache_dtype,
                    chunks=score_chunks,
                    compressor=compressor,
                )

    def write_batch(
        self,
        start: int,
        *,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        split_ids: np.ndarray,
        sequence_indices: np.ndarray,
        residuals: Mapping[int, np.ndarray],
        attention_scores: Mapping[int, np.ndarray] | None = None,
    ) -> None:
        input_ids = np.asarray(input_ids, dtype=np.int32)
        batch_size = int(input_ids.shape[0])
        stop = start + batch_size
        if start < 0 or stop > self.spec.num_samples:
            raise IndexError(f"Batch [{start}:{stop}] exceeds shard size {self.spec.num_samples}")
        expected_2d = (batch_size, self.spec.sequence_length)
        if input_ids.shape != expected_2d:
            raise ValueError(f"input_ids must have shape {expected_2d}, got {input_ids.shape}")

        self._root["input_ids"][start:stop] = input_ids
        self._root["attention_mask"][start:stop] = np.asarray(attention_mask, dtype=np.uint8)
        self._root["split_id"][start:stop] = np.asarray(split_ids, dtype=np.uint8)
        self._root["sequence_index"][start:stop] = np.asarray(sequence_indices, dtype=np.int64)

        expected_3d = (batch_size, self.spec.sequence_length, self.spec.d_model)
        for layer_idx in self.spec.layer_indices:
            if layer_idx not in residuals:
                raise KeyError(f"Missing residuals for layer {layer_idx}")
            array = np.asarray(residuals[layer_idx])
            if array.shape != expected_3d:
                raise ValueError(
                    f"Residual layer {layer_idx} must have shape {expected_3d}, got {array.shape}"
                )
            self._root["activations"][f"layer_{layer_idx}"][start:stop] = array

        attention_scores = attention_scores or {}
        expected_scores = (batch_size, self.spec.sequence_length, self.spec.sequence_length)
        for layer_idx in self.spec.attention_layers:
            if layer_idx not in attention_scores:
                raise KeyError(f"Missing attention scores for layer {layer_idx}")
            array = np.asarray(attention_scores[layer_idx])
            if array.shape != expected_scores:
                raise ValueError(
                    f"Attention layer {layer_idx} must have shape {expected_scores}, got {array.shape}"
                )
            self._root["attention_scores"][f"attention_layer_{layer_idx}"][start:stop] = array

        self._written[start:stop] = True

    def close(self) -> None:
        if self._store is None:
            return
        if not bool(self._written.all()):
            missing = np.flatnonzero(~self._written).tolist()
            self._store.close()
            self._store = None
            raise RuntimeError(f"Activation shard closed before samples were written: {missing[:10]}")
        self._root["complete"][0] = 1
        self._store.close()
        self._store = None

    def __enter__(self) -> "ActivationShardWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            if self._store is not None:
                self._store.close()
                self._store = None
            return
        self.close()


__all__ = [
    "ActivationShardWriter",
    "ActivationStoreSpec",
    "SPLIT_IDS",
    "SPLIT_NAMES",
]
