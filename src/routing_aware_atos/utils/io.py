from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

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


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cached_samples(path: str | Path) -> List[CachedSample]:
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


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def save_cached_samples(path: str | Path, samples: List[CachedSample]) -> None:
    payload = []
    for sample in samples:
        item = {
            "tokens": list(sample.tokens),
            "residuals": {str(k): np.asarray(v, dtype=np.float32).tolist() for k, v in sample.residuals.items()},
            "attention_scores": {
                f"{k[0]},{k[1]}": np.asarray(v, dtype=np.float32).tolist()
                for k, v in (sample.attention_scores or {}).items()
            },
            "attribution_scores": {
                f"{k[0]},{k[1]}": np.asarray(v, dtype=np.float32).tolist()
                for k, v in (sample.attribution_scores or {}).items()
            },
            "metadata": sample.metadata or {},
        }
        payload.append(item)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
