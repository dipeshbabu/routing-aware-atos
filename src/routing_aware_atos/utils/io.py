from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml

from routing_aware_atos.utils.types import CachedSample


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_cached_samples(path: str | Path) -> List[CachedSample]:
    """Load a simple JSON cache format for demos and tests.

    Format:
    [
      {
        "tokens": [...],
        "residuals": {"10": [[...]], "12": [[...]]},
        "attention_scores": {"10,12": [[...]]},
        "attribution_scores": {"10,12": [[...]]}
      }
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        residuals = {int(k): np.asarray(v, dtype=np.float32) for k, v in item["residuals"].items()}
        attention = {
            tuple(int(x) for x in k.split(",")): np.asarray(v, dtype=np.float32)
            for k, v in item.get("attention_scores", {}).items()
        }
        attribution = {
            tuple(int(x) for x in k.split(",")): np.asarray(v, dtype=np.float32)
            for k, v in item.get("attribution_scores", {}).items()
        }
        samples.append(
            CachedSample(
                tokens=item["tokens"],
                residuals=residuals,
                attention_scores=attention or None,
                attribution_scores=attribution or None,
                metadata=item.get("metadata"),
            )
        )
    return samples
