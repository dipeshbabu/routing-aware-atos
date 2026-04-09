from __future__ import annotations

import numpy as np

from routing_aware_atos.evaluation.policy_comparison import compare_policy_runs
from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig
from routing_aware_atos.utils.io import save_npz


def test_compare_policy_runs(tmp_path):
    rng = np.random.default_rng(2)
    decoder = rng.normal(size=(4, 3)).astype(np.float32)
    decoder_path = tmp_path / 'decoder.npz'
    save_npz(decoder_path, decoder=decoder)

    runs = []
    for idx, noise in enumerate([0.0, 0.1]):
        X = rng.normal(size=(25, 3)).astype(np.float32)
        W = rng.normal(size=(3, 3)).astype(np.float32)
        Y = X @ W + noise * rng.normal(size=(25, 3)).astype(np.float32)
        pair_path = tmp_path / f'pairs_{idx}.npz'
        op_path = tmp_path / f'op_{idx}.npz'
        save_npz(pair_path, X=X, Y=Y)
        TransportOperator(TransportOperatorConfig(ridge_lambda=1e-6)).fit(X, Y).save(op_path)
        runs.append({
            'policy_name': f'policy_{idx}',
            'rank': None,
            'operator_path': str(op_path),
            'pairs_path': str(pair_path),
            'decoder_path': str(decoder_path),
        })

    payload = compare_policy_runs(runs)
    assert len(payload['summary_rows']) == 2
    assert payload['summary_rows'][0]['mean_r2'] <= 1.0
