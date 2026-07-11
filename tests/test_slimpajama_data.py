from __future__ import annotations

from pathlib import Path

import numpy as np

from routing_aware_atos.activation_loader import ActivationLoader
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
    split_token_budgets,
)
from scripts.collect_hf_activations import (
    _completed_shard_matches,
    _file_sha256,
    _manifest_shards_match,
)


class _Tokenizer:
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {"input_ids": [3 + ord(char) % 13 for char in text]}


def test_pack_token_budget_is_exact_and_pads_only_final_sequence():
    rows = [{"text": "abcdef"}, {"text": "ghijklmnop"}]
    documents = iter_document_token_ids(rows, _Tokenizer())
    packed = pack_token_budget(
        documents,
        token_budget=11,
        sequence_length=4,
        bos_token_id=1,
        pad_token_id=0,
        split="train",
        source_split="train",
    )

    assert [sample.valid_tokens for sample in packed] == [4, 4, 3]
    assert sum(sample.valid_tokens for sample in packed) == 11
    assert all(sample.input_ids[0] == 1 for sample in packed)
    assert packed[-1].attention_mask.tolist() == [1, 1, 1, 0]


def test_split_token_budgets_have_no_rounding_loss():
    budgets = split_token_budgets(
        250_000,
        {"train": 0.6, "validation": 0.2, "test": 0.2},
    )
    assert budgets == {"train": 150_000, "validation": 50_000, "test": 50_000}


def test_document_collection_does_not_join_documents():
    documents = iter([[10, 11, 2], [20, 21, 22, 2]])
    sequences = collect_document_token_budget(
        documents,
        token_budget=8,
        sequence_length=5,
        bos_token_id=1,
        pad_token_id=0,
        split="train",
        source_split="train",
    )
    assert sequences[0].input_ids.tolist() == [1, 10, 11, 2, 0]
    assert sequences[1].input_ids.tolist() == [1, 20, 21, 22, 0]


def test_causal_document_collection_skips_short_documents():
    documents = iter([[10], [20, 21, 22, 23], [30, 31, 32, 33]])
    sequences = collect_full_document_sequences(
        documents,
        num_sequences=2,
        sequence_length=5,
        bos_token_id=1,
        split="causal",
        source_split="test",
    )
    assert [sequence.input_ids[1] for sequence in sequences] == [20, 30]
    assert all(sequence.valid_tokens == 5 for sequence in sequences)


def _write_shard(path: Path, split_names: list[str], offset: int) -> None:
    n = len(split_names)
    spec = ActivationStoreSpec(
        num_samples=n,
        sequence_length=3,
        d_model=2,
        layer_indices=(1, 2),
        attention_layers=(2,),
        cache_dtype="float32",
    )
    input_ids = np.arange(offset, offset + n * 3, dtype=np.int32).reshape(n, 3)
    mask = np.ones((n, 3), dtype=np.uint8)
    residuals = {
        1: np.arange(n * 3 * 2, dtype=np.float32).reshape(n, 3, 2) + offset,
        2: np.arange(n * 3 * 2, dtype=np.float32).reshape(n, 3, 2) + offset + 1,
    }
    attention = np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy()
    with ActivationShardWriter(path, spec) as writer:
        writer.write_batch(
            0,
            input_ids=input_ids,
            attention_mask=mask,
            split_ids=np.asarray([SPLIT_IDS[name] for name in split_names]),
            sequence_indices=np.arange(offset, offset + n),
            residuals=residuals,
            attention_scores={2: attention},
        )


def test_activation_shards_support_variable_sizes_splits_and_dynamic_attribution(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_shard(cache_dir / "part-00000.zip", ["train", "validation"], 0)
    _write_shard(cache_dir / "part-00001.zip", ["test"], 10)

    loader = ActivationLoader(activation_dir_path=cache_dir)
    assert len(loader) == 3
    assert loader.sample_map(2) == (1, 0)
    assert loader.indices_for_split("train") == [0]
    assert loader.indices_for_split("validation") == [1]
    assert loader.indices_for_split("test") == [2]

    sample = loader.get_cached_sample(
        2,
        layer_indices=[1, 2],
        attribution_layer_pairs=[(1, 2)],
    )
    assert sample.metadata["split"] == "test"
    assert sample.metadata["sequence_index"] == 10
    assert sample.attribution_scores is not None
    assert sample.attribution_scores[(1, 2)].shape == (3, 3)
    assert np.allclose(sample.attribution_scores[(1, 2)].sum(axis=1), 1.0)
    loader.close()


def test_completed_activation_shard_is_verified_before_resume(tmp_path: Path):
    path = tmp_path / "part-00000.zip"
    sequence = PackedSequence(
        input_ids=np.asarray([1, 2, 3], dtype=np.int32),
        attention_mask=np.ones(3, dtype=np.uint8),
        split="train",
        sequence_index=4,
        source_split="train",
    )
    spec = ActivationStoreSpec(
        num_samples=1,
        sequence_length=3,
        d_model=2,
        layer_indices=(1,),
        cache_dtype="float32",
    )
    with ActivationShardWriter(path, spec) as writer:
        writer.write_batch(
            0,
            input_ids=sequence.input_ids[None, :],
            attention_mask=sequence.attention_mask[None, :],
            split_ids=np.asarray([SPLIT_IDS["train"]]),
            sequence_indices=np.asarray([sequence.sequence_index]),
            residuals={1: np.ones((1, 3, 2), dtype=np.float32)},
        )

    assert _completed_shard_matches(path, spec, [sequence])
    changed_sequence = PackedSequence(
        input_ids=sequence.input_ids,
        attention_mask=sequence.attention_mask,
        split=sequence.split,
        sequence_index=99,
        source_split=sequence.source_split,
    )
    assert not _completed_shard_matches(path, spec, [changed_sequence])


def test_activation_manifest_detects_shard_corruption(tmp_path: Path):
    shard = tmp_path / "part-00000.zip"
    shard.write_bytes(b"verified activation bytes")
    manifest = {
        "shards": [
            {
                "path": shard.name,
                "sha256": _file_sha256(shard),
                "num_samples": 1,
            }
        ]
    }

    assert _manifest_shards_match(tmp_path, manifest)
    shard.write_bytes(b"corrupted activation bytes")
    assert not _manifest_shards_match(tmp_path, manifest)
