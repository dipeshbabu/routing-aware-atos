from __future__ import annotations

import numpy as np
import torch

from routing_aware_atos.causal_eval.live_restore import evaluate_live_causal_restoration


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
    )

    assert metrics["ablated_logit_mse"] > 0
    assert metrics["restored_logit_mse"] < 1e-12
    assert metrics["logit_mse_restoration"] > 0.99


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
