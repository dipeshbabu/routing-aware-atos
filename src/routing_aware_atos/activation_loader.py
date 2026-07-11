from __future__ import annotations

import logging
import os
from bisect import bisect_right
from pathlib import Path
from typing import Generator, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from routing_aware_atos.routed_types import CachedSample, LayerPair
from routing_aware_atos.utils.io import load_cached_samples

try:
    import zarr
except ImportError:  # pragma: no cover - optional dependency
    zarr = None

logger = logging.getLogger(__name__)


class ActivationLoader:
    """Load routing-ready samples from zarr activation artifacts, JSON caches, or memory."""

    def __init__(
        self,
        activation_dir_path: str | Path | None = None,
        samples_path: str | Path | None = None,
        *,
        samples: Sequence[CachedSample] | None = None,
    ):
        provided = sum(value is not None for value in (activation_dir_path, samples_path, samples))
        if provided != 1:
            raise ValueError("Provide exactly one of activation_dir_path, samples_path, or samples")

        self.activation_dir_path = Path(activation_dir_path) if activation_dir_path is not None else None
        self.samples_path = Path(samples_path) if samples_path is not None else None
        self._samples = list(samples or []) if samples is not None else None

        self.store_objects: dict[int, object] = {}
        self.root_objects: dict[int, object] = {}
        self.file_paths: list[Path] = []
        self.sample_offsets: list[int] = [0]
        self.num_samples = 0
        self.samples_per_file = 0
        self._backend = "memory"

        if self.activation_dir_path is not None:
            if zarr is None:
                raise ImportError("zarr is required to load activation artifacts")
            if not self.activation_dir_path.exists():
                raise ValueError(f"Activation directory {self.activation_dir_path} does not exist.")
            self._backend = "zarr"
            self.create_store_objects()
        elif self.samples_path is not None:
            self._backend = "json"
            self._samples = load_cached_samples(self.samples_path)
        else:
            self._backend = "memory"

    def __len__(self) -> int:
        if self._backend == "zarr":
            return self.num_samples
        return len(self._samples or [])

    def _require_zarr_backend(self) -> None:
        if self._backend != "zarr":
            raise ValueError("This method requires an activation_dir_path-backed zarr loader")

    def sample_map(self, idx: int) -> tuple[int, int]:
        self._require_zarr_backend()
        if not self.file_paths:
            raise ValueError("Sample map is not created.")
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"sample_idx={idx} outside [0, {self.num_samples - 1}]")
        part_id = bisect_right(self.sample_offsets, idx) - 1
        return part_id, idx - self.sample_offsets[part_id]

    def _get_file_list(self) -> list[str]:
        self._require_zarr_backend()
        assert self.activation_dir_path is not None
        entries = []
        for name in sorted(os.listdir(self.activation_dir_path)):
            path = self.activation_dir_path / name
            if path.is_file() and path.suffix.lower() == ".zip":
                entries.append(name)
            elif path.is_dir() and (
                path.suffix.lower() == ".zarr"
                or (path / "zarr.json").exists()
                or (path / ".zgroup").exists()
            ):
                entries.append(name)
        return entries

    def create_store_objects(self) -> None:
        self._require_zarr_backend()
        assert zarr is not None
        assert self.activation_dir_path is not None

        file_names = self._get_file_list()
        if not file_names:
            raise ValueError(f"No .zip or .zarr activation shards found in {self.activation_dir_path}")

        for i, file_name in enumerate(file_names):
            file_path = self.activation_dir_path / file_name
            if file_path.is_dir():
                store = None
                root = zarr.open_group(str(file_path), mode="r")
            else:
                store = zarr.storage.ZipStore(file_path, mode="r")
                root = zarr.open_group(store=store, mode="r")
                self.store_objects[i] = store
            self.root_objects[i] = root
            self.file_paths.append(file_path)
            shard_samples = int(root["input_ids"].shape[0])
            self.num_samples += shard_samples
            self.sample_offsets.append(self.num_samples)

        if len(self.root_objects) != len(file_names):
            raise ValueError("Not all zarr activation files were loaded.")
        self.samples_per_file = int(self.root_objects[0]["input_ids"].shape[0])

    def close(self) -> None:
        for store in self.store_objects.values():
            close = getattr(store, "close", None)
            if close is not None:
                close()
        self.store_objects.clear()
        self.root_objects.clear()

    def __enter__(self) -> "ActivationLoader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _get_root(self, sample_idx: int):
        part_id, local_sample_id = self.sample_map(sample_idx)
        return self.root_objects[part_id], local_sample_id

    def get_sample_sequence_length(self, sample_idx: int) -> int:
        if self._backend == "zarr":
            z, local_sample_id = self._get_root(sample_idx)
            return int(np.asarray(z["attention_mask"][local_sample_id], dtype=np.int32).sum())

        sample = self.load_cached_sample(sample_idx)
        return sample.seq_len

    def get_input_ids(self, sample_idx: int) -> np.ndarray:
        """
        Return the padded input_ids row for a given sample.

        Shape:
            [seq_len]
        """
        self._require_zarr_backend()
        z, local_sample_id = self._get_root(sample_idx)
        return np.asarray(z["input_ids"][local_sample_id], dtype=np.int32)

    def get_attention_mask(self, sample_idx: int) -> np.ndarray:
        """
        Return the padded attention mask row for a given sample.

        Shape:
            [seq_len]
        """
        self._require_zarr_backend()
        z, local_sample_id = self._get_root(sample_idx)
        return np.asarray(z["attention_mask"][local_sample_id], dtype=np.int32)

    def get_sample_split(self, sample_idx: int) -> str | None:
        if self._backend == "zarr":
            z, local_sample_id = self._get_root(sample_idx)
            if "split_id" not in z:
                return None
            split_id = int(np.asarray(z["split_id"][local_sample_id]))
            split_names = dict(z.attrs.get("split_names", {}))
            return split_names.get(str(split_id), split_names.get(split_id))

        sample = self.load_cached_sample(sample_idx)
        split = (sample.metadata or {}).get("split")
        return None if split is None else str(split)

    def indices_for_split(self, split_name: str) -> list[int]:
        split_name = str(split_name)
        indices = [idx for idx in range(len(self)) if self.get_sample_split(idx) == split_name]
        if not indices:
            available = sorted(
                {name for idx in range(len(self)) if (name := self.get_sample_split(idx)) is not None}
            )
            raise ValueError(f"No samples found for split {split_name!r}; available splits: {available}")
        return indices

    def get_sequence_input_ids(self, sample_idx: int) -> np.ndarray:
        """
        Return unpadded input_ids for a given sample.

        Shape:
            [valid_seq_len]
        """
        if self._backend == "zarr":
            input_ids = self.get_input_ids(sample_idx)
            attention_mask = self.get_attention_mask(sample_idx)
            valid_len = int(attention_mask.sum())
            return np.asarray(input_ids[:valid_len], dtype=np.int32)

        sample = self.load_cached_sample(sample_idx)
        return np.asarray(sample.tokens, dtype=np.int32)

    def get_layer_residuals(
        self,
        sample_idx: int,
        layer_indices: list[int],
    ) -> dict[int, np.ndarray]:
        """
        Return unpadded residual activations for the requested layers.

        Returns:
            dict[layer_idx] -> np.ndarray of shape [valid_seq_len, d_model]
        """
        if self._backend == "zarr":
            z, local_sample_id = self._get_root(sample_idx)

            attention_mask = np.asarray(z["attention_mask"][local_sample_id], dtype=np.int32)
            valid_len = int(attention_mask.sum())

            residuals: dict[int, np.ndarray] = {}
            for layer_idx in layer_indices:
                arr = np.asarray(
                    z["activations"][f"layer_{layer_idx}"][local_sample_id, :valid_len, :],
                    dtype=np.float32,
                )
                residuals[layer_idx] = arr

            return residuals

        sample = self.load_cached_sample(sample_idx, layers=layer_indices)
        return {int(k): np.asarray(v, dtype=np.float32) for k, v in sample.residuals.items()}

    def _try_load_score_matrix(
        self,
        z: object,
        group_name: str,
        array_name: str,
        local_sample_id: int,
        valid_len: int,
    ) -> Optional[np.ndarray]:
        """
        Try to load a precomputed routing score matrix from the zarr file.

        Expected optional layout:
            attention_scores/attention_layer_{layer_idx}
            attribution_scores/attribution_layer_{layer_idx}

        with shape:
            [num_samples, seq_len, seq_len]

        Returns:
            np.ndarray of shape [valid_len, valid_len], or None if not found.
        """
        if group_name not in z:
            return None

        group = z[group_name]
        if array_name not in group:
            return None

        matrix = np.asarray(
            group[array_name][local_sample_id, :valid_len, :valid_len],
            dtype=np.float32,
        )
        return matrix

    def get_attention_scores(
        self,
        sample_idx: int,
        source_layer: int,
        target_layer: int,
    ) -> Optional[np.ndarray]:
        """
        Return a routing score matrix for attention-based routing.

        Shape:
            [target_seq_len, source_seq_len]

        Notes:
            - The current artifact stores one pooled attention matrix per target layer:
                attention_scores/attention_layer_{target_layer}
            - source_layer is kept in the signature for compatibility with routing code,
              but is not used by the current storage layout.
        """
        if self._backend == "zarr":
            z, local_sample_id = self._get_root(sample_idx)

            attention_mask = np.asarray(z["attention_mask"][local_sample_id], dtype=np.int32)
            valid_len = int(attention_mask.sum())

            array_name = f"attention_layer_{target_layer}"
            return self._try_load_score_matrix(
                z=z,
                group_name="attention_scores",
                array_name=array_name,
                local_sample_id=local_sample_id,
                valid_len=valid_len,
            )

        sample = self.load_cached_sample(sample_idx, attention_layer_pairs=[(source_layer, target_layer)])
        if sample.attention_scores is None:
            return None
        return np.asarray(sample.attention_scores[(source_layer, target_layer)], dtype=np.float32)

    def get_attribution_scores(
        self,
        sample_idx: int,
        source_layer: int,
        target_layer: int,
    ) -> Optional[np.ndarray]:
        """
        Return a routing score matrix for attribution-based routing.

        Shape:
            [target_seq_len, source_seq_len]

        Notes:
            - The current expected layout is:
                attribution_scores/attribution_layer_{target_layer}
            - source_layer is kept in the signature for compatibility with routing code,
              but is not used by the current storage layout.
        """
        if self._backend == "zarr":
            z, local_sample_id = self._get_root(sample_idx)

            attention_mask = np.asarray(z["attention_mask"][local_sample_id], dtype=np.int32)
            valid_len = int(attention_mask.sum())

            exact_name = f"attribution_layer_{source_layer}_to_{target_layer}"
            matrix = self._try_load_score_matrix(
                z=z,
                group_name="attribution_scores",
                array_name=exact_name,
                local_sample_id=local_sample_id,
                valid_len=valid_len,
            )
            if matrix is not None:
                return matrix
            return self._try_load_score_matrix(
                z=z,
                group_name="attribution_scores",
                array_name=f"attribution_layer_{target_layer}",
                local_sample_id=local_sample_id,
                valid_len=valid_len,
            )

        sample = self.load_cached_sample(sample_idx, attribution_layer_pairs=[(source_layer, target_layer)])
        if sample.attribution_scores is None:
            return None
        return np.asarray(sample.attribution_scores[(source_layer, target_layer)], dtype=np.float32)

    def list_available_routing_scores(self, sample_idx: int = 0) -> dict[str, list[str]]:
        """
        Inspect which routing score arrays are present in a stored activation artifact.

        Returns:
            {
                "attention_scores": [...],
                "attribution_scores": [...],
            }
        """
        if self._backend == "zarr":
            z, _ = self._get_root(sample_idx)

            out = {
                "attention_scores": [],
                "attribution_scores": [],
            }

            if "attention_scores" in z:
                out["attention_scores"] = sorted(list(z["attention_scores"].array_keys()))

            if "attribution_scores" in z:
                out["attribution_scores"] = sorted(list(z["attribution_scores"].array_keys()))

            return out

        sample = (self._samples or [])[sample_idx]
        return {
            "attention_scores": sorted(
                [f"attention_layer_{key[1]}" for key in (sample.attention_scores or {}).keys()]
            ),
            "attribution_scores": sorted(
                [f"attribution_layer_{key[1]}" for key in (sample.attribution_scores or {}).keys()]
            ),
        }

    def get_cached_sample(
        self,
        sample_idx: int,
        layer_indices: list[int],
        attention_layer_pairs: Optional[list[tuple[int, int]]] = None,
        attribution_layer_pairs: Optional[list[tuple[int, int]]] = None,
    ) -> CachedSample:
        """
        Build a routing-ready CachedSample from the activation artifact.
        """
        if self._backend == "zarr":
            tokens = self.get_sequence_input_ids(sample_idx)
            residuals = self.get_layer_residuals(sample_idx, layer_indices)

            attention_scores = None
            requested_attention = set(attention_layer_pairs or [])
            requested_attention.update(attribution_layer_pairs or [])
            if requested_attention:
                attention_scores = {}
                for source_layer, target_layer in sorted(requested_attention):
                    matrix = self.get_attention_scores(
                        sample_idx=sample_idx,
                        source_layer=source_layer,
                        target_layer=target_layer,
                    )
                    if matrix is not None:
                        # Current storage layout indexes pooled routing matrices by target layer only.
                        # We still store them under (source_layer, target_layer) for compatibility
                        # with routing-aware dataset builders.
                        attention_scores[(source_layer, target_layer)] = matrix

            attribution_scores = None
            if attribution_layer_pairs:
                attribution_scores = {}
                for source_layer, target_layer in attribution_layer_pairs:
                    matrix = self.get_attribution_scores(
                        sample_idx=sample_idx,
                        source_layer=source_layer,
                        target_layer=target_layer,
                    )
                    if matrix is not None:
                        attribution_scores[(source_layer, target_layer)] = matrix

            z, local_sample_id = self._get_root(sample_idx)
            split_name = self.get_sample_split(sample_idx)
            sequence_index = (
                int(np.asarray(z["sequence_index"][local_sample_id]))
                if "sequence_index" in z
                else int(sample_idx)
            )

            sample = CachedSample(
                tokens=tokens,
                residuals=residuals,
                attention_scores=attention_scores or None,
                attribution_scores=attribution_scores or None,
                metadata={
                    "sample_idx": int(sample_idx),
                    "sequence_index": sequence_index,
                    "sequence_length": int(len(tokens)),
                    "split": split_name,
                    "layer_semantics": str(z.attrs.get("layer_semantics", "unknown")),
                },
            )
            if attribution_layer_pairs:
                from routing_aware_atos.attribution_builder import build_attribution_score_matrix

                generated = dict(sample.attribution_scores or {})
                for source_layer, target_layer in attribution_layer_pairs:
                    key = (source_layer, target_layer)
                    if key not in generated:
                        generated[key] = build_attribution_score_matrix(
                            sample,
                            source_layer,
                            target_layer,
                            method="attention_value",
                        )
                sample.attribution_scores = generated
            sample.validate()
            return sample

        sample = self.load_cached_sample(
            sample_idx,
            layers=layer_indices,
            attention_layer_pairs=attention_layer_pairs,
            attribution_layer_pairs=attribution_layer_pairs,
        )
        metadata = dict(sample.metadata or {})
        metadata.setdefault("sample_idx", int(sample_idx))
        metadata.setdefault("sequence_length", int(sample.seq_len))
        loaded = CachedSample(
            tokens=sample.tokens,
            residuals=sample.residuals,
            attention_scores=sample.attention_scores,
            attribution_scores=sample.attribution_scores,
            metadata=metadata,
        )
        loaded.validate()
        return loaded

    def get_cached_sample_for_pair(
        self,
        sample_idx: int,
        source_layer: int,
        target_layer: int,
        include_attention: bool = True,
        include_attribution: bool = False,
    ) -> CachedSample:
        """
        Convenience wrapper for the common case of one source-target layer pair.
        """
        return self.get_cached_sample(
            sample_idx=sample_idx,
            layer_indices=[source_layer, target_layer],
            attention_layer_pairs=[(source_layer, target_layer)] if include_attention else None,
            attribution_layer_pairs=[(source_layer, target_layer)] if include_attribution else None,
        )

    def load_cached_sample(
        self,
        sample_idx: int,
        *,
        layers: Iterable[int] | None = None,
        attention_layer_pairs: Iterable[LayerPair] | None = None,
        attribution_layer_pairs: Iterable[LayerPair] | None = None,
    ) -> CachedSample:
        if self._backend == "zarr":
            layer_indices = list(layers) if layers is not None else []
            if not layer_indices:
                raise ValueError("layers must be provided when loading from zarr artifacts")
            return self.get_cached_sample(
                sample_idx,
                layer_indices=layer_indices,
                attention_layer_pairs=list(attention_layer_pairs) if attention_layer_pairs is not None else None,
                attribution_layer_pairs=list(attribution_layer_pairs) if attribution_layer_pairs is not None else None,
            )

        sample = (self._samples or [])[sample_idx]
        residuals = self._select_layers(sample.residuals, layers)
        attention_scores = self._select_layer_pairs(sample.attention_scores, attention_layer_pairs)
        attribution_scores = self._select_layer_pairs(sample.attribution_scores, attribution_layer_pairs)
        loaded = CachedSample(
            tokens=sample.tokens,
            residuals=residuals,
            attention_scores=attention_scores,
            attribution_scores=attribution_scores,
            metadata=dict(sample.metadata or {}),
        )
        loaded.validate()
        return loaded

    def iter_cached_samples(
        self,
        idx_list: Optional[list[int]] = None,
        layer_indices: Optional[list[int]] = None,
        attention_layer_pairs: Optional[list[tuple[int, int]]] = None,
        attribution_layer_pairs: Optional[list[tuple[int, int]]] = None,
        *,
        layers: Iterable[int] | None = None,
        split_name: str | None = None,
        strict: bool = False,
    ) -> Generator[CachedSample, None, None]:
        """
        Iterate over routing-ready CachedSample objects.
        """
        if layer_indices is None:
            if layers is not None:
                layer_indices = list(layers)
            elif self._backend == "zarr":
                raise ValueError("layer_indices must be provided")

        if idx_list is None:
            idx_list = self.indices_for_split(split_name) if split_name is not None else list(range(len(self)))
        elif split_name is not None:
            idx_list = [idx for idx in idx_list if self.get_sample_split(idx) == split_name]

        for sample_idx in idx_list:
            try:
                if layer_indices is not None:
                    yield self.get_cached_sample(
                        sample_idx=sample_idx,
                        layer_indices=layer_indices,
                        attention_layer_pairs=attention_layer_pairs,
                        attribution_layer_pairs=attribution_layer_pairs,
                    )
                else:
                    yield self.load_cached_sample(
                        sample_idx,
                        layers=layers,
                        attention_layer_pairs=attention_layer_pairs,
                        attribution_layer_pairs=attribution_layer_pairs,
                    )
            except (ValueError, IndexError, KeyError) as e:
                if strict:
                    raise
                logger.warning(
                    "Skipping cached sample %d due to error: %s",
                    sample_idx,
                    str(e),
                    exc_info=True,
                )
                continue

    def get_attention_matrix(self, sample_idx: int, source_layer: int, target_layer: int) -> np.ndarray:
        matrix = self.get_attention_scores(sample_idx, source_layer, target_layer)
        if matrix is None:
            raise KeyError(f"Missing attention scores for layer pair {(source_layer, target_layer)}")
        return matrix

    def get_attribution_matrix(self, sample_idx: int, source_layer: int, target_layer: int) -> np.ndarray:
        matrix = self.get_attribution_scores(sample_idx, source_layer, target_layer)
        if matrix is None:
            raise KeyError(f"Missing attribution scores for layer pair {(source_layer, target_layer)}")
        return matrix

    @staticmethod
    def _select_layers(
        residuals: Mapping[int, np.ndarray],
        layers: Iterable[int] | None,
    ) -> Mapping[int, np.ndarray]:
        if layers is None:
            return {int(k): np.asarray(v, dtype=np.float32) for k, v in residuals.items()}

        selected = {}
        for layer in layers:
            if layer not in residuals:
                raise KeyError(f"Missing residual layer {layer}")
            selected[int(layer)] = np.asarray(residuals[layer], dtype=np.float32)
        return selected

    @staticmethod
    def _select_layer_pairs(
        scores: Mapping[LayerPair, np.ndarray] | None,
        layer_pairs: Iterable[LayerPair] | None,
    ) -> Mapping[LayerPair, np.ndarray] | None:
        if scores is None:
            return None
        if layer_pairs is None:
            return None

        selected = {}
        for pair in layer_pairs:
            if pair not in scores:
                raise KeyError(f"Missing score matrix for layer pair {pair}")
            selected[tuple(pair)] = np.asarray(scores[pair], dtype=np.float32)
        return selected or None
