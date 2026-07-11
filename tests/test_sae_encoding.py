import numpy as np
import pytest

from routing_aware_atos.sae.encoding import (
    active_feature_mask,
    sae_feature_activations,
    validate_sae_artifact,
)


def test_sae_feature_activations_supports_jumprelu_thresholds():
    arrays = {
        "decoder": np.eye(2, dtype=np.float32),
        "encoder": np.eye(2, dtype=np.float32),
        "b_dec": np.asarray([1.0, 0.0], dtype=np.float32),
        "apply_b_dec_to_input": np.asarray(1, dtype=np.uint8),
        "b_enc": np.asarray([0.0, 0.5], dtype=np.float32),
        "threshold": np.asarray([0.5, 1.0], dtype=np.float32),
    }
    residuals = np.asarray([[2.0, 0.6], [1.2, 1.0]], dtype=np.float32)

    activations = sae_feature_activations(residuals, arrays)
    assert np.allclose(activations, [[1.0, 1.1], [0.0, 1.5]])
    assert active_feature_mask(residuals, arrays).tolist() == [[True, True], [False, True]]


def test_sae_feature_activations_can_select_features():
    arrays = {
        "decoder": np.eye(3, dtype=np.float32),
        "encoder": np.eye(3, dtype=np.float32),
    }
    residuals = np.asarray([[1.0, -1.0, 2.0]], dtype=np.float32)
    assert np.allclose(
        sae_feature_activations(residuals, arrays, feature_ids=[2, 0]),
        [[2.0, 1.0]],
    )


def test_sae_feature_activations_honors_disabled_decoder_bias():
    arrays = {
        "decoder": np.eye(2, dtype=np.float32),
        "encoder": np.eye(2, dtype=np.float32),
        "b_dec": np.asarray([10.0, 10.0], dtype=np.float32),
        "apply_b_dec_to_input": np.asarray(0, dtype=np.uint8),
    }
    residuals = np.asarray([[1.0, 2.0]], dtype=np.float32)
    assert np.allclose(sae_feature_activations(residuals, arrays), residuals)


def test_sae_artifact_rejects_unsupported_activation_function():
    arrays = {
        "decoder": np.eye(2, dtype=np.float32),
        "encoder": np.eye(2, dtype=np.float32),
        "activation_fn": np.asarray("gelu"),
    }
    with pytest.raises(ValueError, match="activation function"):
        validate_sae_artifact(arrays)
