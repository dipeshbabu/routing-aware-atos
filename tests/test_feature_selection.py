import numpy as np
import torch
from types import SimpleNamespace

from routing_aware_atos.sae.feature_selection import (
    binned_entropy,
    composite_feature_scores,
    max_abs_anchor_correlation,
    minmax_normalize,
    normalized_binned_entropy,
    reference_style_feature_scores,
)
from scripts.select_sae_features import (
    _TorchSAEEncoder,
    _causal_effect_scores,
    _compute_logit_focus,
)


def test_normalized_binned_entropy_distinguishes_focused_features():
    counts = np.asarray([[10, 0, 0], [5, 5, 0], [2, 2, 2]], dtype=np.int64)
    entropy = normalized_binned_entropy(counts)
    assert entropy[0] == 0.0
    assert entropy[1] > entropy[0]
    assert entropy[2] >= entropy[1]


def test_binned_entropy_matches_discrete_entropy():
    entropy = binned_entropy(np.asarray([[5, 5], [10, 0]], dtype=np.int64))
    np.testing.assert_allclose(entropy, [np.log(2.0), 0.0], atol=1e-6)


def test_anchor_correlation_uses_centered_pearson_statistics():
    activations = np.asarray(
        [
            [10.0, 1.0, 4.0],
            [11.0, 2.0, 4.0],
            [12.0, 3.0, 4.0],
            [13.0, 4.0, 4.0],
        ]
    )
    centered = activations - activations.mean(axis=0, keepdims=True)
    anchors = np.asarray([0], dtype=np.int64)
    redundancy = max_abs_anchor_correlation(
        centered[:, anchors].T @ centered,
        np.sum(centered**2, axis=0),
        anchors,
    )
    np.testing.assert_allclose(redundancy, [0.0, 1.0, 0.0], atol=1e-6)


def test_composite_feature_scores_rewards_coherence_and_uniqueness():
    scores = composite_feature_scores(
        activation_strength=np.asarray([1.0, 1.0]),
        token_entropy=np.asarray([0.1, 0.9]),
        max_decoder_cosine=np.asarray([0.1, 0.9]),
    )
    assert scores[0] > scores[1]


def test_minmax_normalize_handles_constant_and_non_finite_values():
    np.testing.assert_array_equal(minmax_normalize(np.asarray([2.0, 2.0])), [0.0, 0.0])
    normalized = minmax_normalize(np.asarray([0.0, np.nan, 2.0]))
    np.testing.assert_allclose(normalized, [0.0, 0.0, 1.0])


def test_reference_style_score_rewards_focus_coherence_and_causal_effect():
    scores = reference_style_feature_scores(
        token_entropy=np.asarray([0.1, 0.9, 0.5]),
        vocabulary_focus=np.asarray([3.0, 1.0, 2.0]),
        redundancy=np.asarray([0.1, 0.9, 0.5]),
        activation_rate=np.asarray([0.05, 0.001, 0.02]),
        causal_effect=np.asarray([0.8, 0.0, 0.2]),
    )
    assert scores[0] > scores[2] > scores[1]


def test_reference_style_pre_score_does_not_require_causal_effect():
    scores = reference_style_feature_scores(
        token_entropy=np.asarray([0.1, 0.9]),
        vocabulary_focus=np.asarray([2.0, 1.0]),
        redundancy=np.asarray([0.1, 0.8]),
        activation_rate=np.asarray([0.05, 0.001]),
    )
    assert scores[0] > scores[1]


class _ToyLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer()])


class _ToyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(6, 2)
        self.model = _ToyBackbone()
        self.lm_head = torch.nn.Linear(2, 6, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


class _ToyTokenizer:
    def __call__(self, prompts, **kwargs):
        batch_size = len(prompts)
        return {
            "input_ids": torch.tensor([[0, 1, 2, 3]] * batch_size),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }


def test_reference_feature_model_metrics_run_live_and_remove_hook():
    torch.manual_seed(0)
    model = _ToyLanguageModel()
    decoder = np.eye(2, dtype=np.float32)
    encoder = _TorchSAEEncoder(
        {"decoder": decoder, "encoder": decoder.T},
        "cpu",
    )

    focus, entropy = _compute_logit_focus(
        decoder,
        np.asarray([0, 1]),
        output_embedding=model.get_output_embeddings().weight,
        device="cpu",
        batch_size=1,
    )
    effects = _causal_effect_scores(
        model,
        _ToyTokenizer(),
        layer=0,
        candidate_ids=np.asarray([0]),
        decoder=decoder,
        encoder=encoder,
        prompts=["probe"],
        device="cpu",
    )

    assert focus.shape == entropy.shape == (2,)
    assert effects.shape == (1,)
    assert np.isfinite(effects).all()
    assert not model.model.layers[0]._forward_hooks
