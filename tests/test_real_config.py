from pathlib import Path

from routing_aware_atos.utils.io import load_yaml
from scripts.preflight_real_experiment import _check_config


def test_paper_scale_real_config_is_internally_consistent():
    config_path = Path("configs/real/gemma2_2b_slimpajama.yaml")
    cfg = load_yaml(config_path)
    errors: list[str] = []

    _check_config(cfg, errors)

    assert errors == []
    assert cfg["dataset"]["name"] == "cerebras/SlimPajama-627B"
    assert cfg["collection"]["operator_tokens"] == 250_000
    assert cfg["collection"]["cache_dtype"] == "float32"
    assert cfg["collection"]["packing_mode"] == "documents"
    assert cfg["feature_selection"]["method"] == "reference_full"
    assert cfg["experiments"]["causal_only"] is True
    pairs = {
        (int(pair["target_layer"]), int(pair["leap"]))
        for pair in cfg["experiments"]["layer_pairs"]
    }
    assert pairs == {(target, leap) for target in (10, 20) for leap in range(1, 11)}
