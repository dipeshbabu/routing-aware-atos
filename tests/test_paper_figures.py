from __future__ import annotations

from pathlib import Path

from routing_aware_atos.evaluation.paper_figures import generate_paper_figures
from routing_aware_atos.utils.io import save_json


def test_generate_paper_figures(tmp_path: Path):
    policy = {"summary_rows": [{"policy_name": "same_token", "rank": None, "mean_r2": 0.5}]}
    causal = {"summary_rows": [{"policy_name": "same_token", "rank": None, "feature_mse_restoration": 0.5}]}
    taxonomy = {"summary_rows": [{"label": "same_token_transport", "count": 2, "fraction": 1.0}]}

    policy_path = tmp_path / "policy.json"
    causal_path = tmp_path / "causal.json"
    taxonomy_path = tmp_path / "taxonomy.json"
    save_json(policy_path, policy)
    save_json(causal_path, causal)
    save_json(taxonomy_path, taxonomy)

    written = generate_paper_figures(
        {
            "output_dir": str(tmp_path / "figs"),
            "policy_comparison_path": str(policy_path),
            "causal_comparison_path": str(causal_path),
            "taxonomy_path": str(taxonomy_path),
        }
    )
    assert "policy_comparison" in written
    assert (tmp_path / "figs" / "figure_policy_comparison.png").exists()
