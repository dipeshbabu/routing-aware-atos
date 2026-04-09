from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from routing_aware_atos.transport_taxonomy import (
    build_feature_policy_matrix,
    build_transport_taxonomy,
    classify_feature_transport,
)
from routing_aware_atos.utils.io import load_json, save_json


def save_transport_taxonomy(payload: Mapping[str, Any], output_path: str | Path) -> None:
    save_json(output_path, payload)


def load_feature_eval_payload(path: str | Path) -> Dict[str, Any]:
    payload = load_json(path)
    if "feature_metrics" not in payload:
        raise ValueError(f"Expected feature evaluation payload at {path}")
    return payload
