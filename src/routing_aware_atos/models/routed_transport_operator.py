
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig
from routing_aware_atos.utils.io import save_json


@dataclass
class RoutedTransportOperator(TransportOperator):
    routing_policy_name: str = "unknown"
    route_summary: Dict[str, Any] = field(default_factory=dict)

    def save_bundle(self, output_dir: str | Path, *, extra_metadata: Dict[str, Any] | None = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.save(output_dir / "operator.npz")
        payload = self.metadata()
        payload["routing_policy_name"] = self.routing_policy_name
        payload["route_summary"] = self.route_summary
        if extra_metadata:
            payload.update(extra_metadata)
        save_json(output_dir / "metadata.json", payload)
