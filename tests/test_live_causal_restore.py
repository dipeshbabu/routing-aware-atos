from __future__ import annotations

import numpy as np
import torch

from routing_aware_atos.causal_eval.live_restore import (
    evaluate_live_causal_restoration,
    logit_mse,
    next_token_cross_entropy,
)


class DummyBlock(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 7, d_model: int = 5):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.block = DummyBlock()
        self.readout = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        hidden = self.block(hidden)
        return self.readout(hidden)


def test_evaluate_live_causal_restoration_restores_logits():
    torch.manual_seed(0)
    model = DummyModel()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        clean_hidden = model.embed(input_ids)[0].detach().cpu().numpy().astype(np.float32)

    metrics = evaluate_live_causal_restoration(
        model,
        input_ids,
        target_layer="block",
        patch_lookup={0: clean_hidden},
        sample_idx_lookup=[0],
        null_patch_lookup={0: np.zeros_like(clean_hidden)},
    )

    assert metrics["ablated_logit_mse"] > 0
    assert metrics["restored_logit_mse"] < 1e-12
    assert metrics["logit_mse_restoration"] > 0.99
    assert "null_cross_entropy" in metrics


def test_evaluate_live_causal_restoration_restores_training_mode_on_error():
    model = DummyModel()
    model.train()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    try:
        evaluate_live_causal_restoration(
            model,
            input_ids,
            target_layer="missing_block",
            patch_lookup={0: np.zeros((4, 5), dtype=np.float32)},
            sample_idx_lookup=[0],
        )
    except AttributeError:
        pass
    else:
        raise AssertionError("invalid target layer should fail")

    assert model.training is True


def test_position_specific_live_restore_changes_only_requested_position():
    torch.manual_seed(0)
    model = DummyModel()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        clean_hidden = model.embed(input_ids)[0].detach().cpu().numpy().astype(np.float32)

    metrics = evaluate_live_causal_restoration(
        model,
        input_ids,
        target_layer="block",
        patch_lookup={7: clean_hidden},
        sample_idx_lookup=[7],
        position_lookup={7: [2]},
    )

    assert metrics["ablated_logit_mse"] > 0
    assert metrics["restored_logit_mse"] < 1e-12


def test_live_metrics_ignore_padded_tokens():
    logits = torch.zeros((1, 4, 5), dtype=torch.float32)
    candidate = logits.clone()
    candidate[:, 2:, :] = 100.0
    input_ids = torch.tensor([[1, 2, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    expected_ce = torch.nn.functional.cross_entropy(logits[:, 0, :], input_ids[:, 1])
    assert np.isclose(
        next_token_cross_entropy(logits, input_ids, attention_mask),
        float(expected_ce),
    )
    assert logit_mse(logits, candidate, attention_mask) == 0.0
