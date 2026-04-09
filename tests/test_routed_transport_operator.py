
from pathlib import Path

from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.data.routed_dataset import RoutedActivationDataset, summarize_routes
from routing_aware_atos.models.routed_transport_operator import RoutedTransportOperator
from routing_aware_atos.models.transport_operator import TransportOperatorConfig
from routing_aware_atos.routing.factory import build_routing_policy


def test_routed_operator_saves_bundle(tmp_path: Path):
    samples = make_mock_samples(num_samples=2, seq_len=5, d_model=3)
    policy = build_routing_policy("attention_topk", top_k=2)
    ds = RoutedActivationDataset(samples=samples, source_layer=10, target_layer=12, routing_policy=policy)
    pairs = ds.build_pairs()

    op = RoutedTransportOperator(
        config=TransportOperatorConfig(ridge_lambda=1e-4, rank=2),
        routing_policy_name=policy.name,
        route_summary=summarize_routes(pairs.routes),
    ).fit(pairs.X, pairs.Y)

    out_dir = tmp_path / "bundle"
    op.save_bundle(out_dir)
    assert (out_dir / "operator.npz").exists()
    assert (out_dir / "metadata.json").exists()
