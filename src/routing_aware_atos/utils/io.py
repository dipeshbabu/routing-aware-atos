from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import numpy as np
import yaml

from routing_aware_atos.utils.types import CachedSample


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp.npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez(temporary_path, **arrays)
        with temporary_path.open("rb+") as temporary:
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, allow_nan=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
    with np.load(path, allow_pickle=False) as data:
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
    save_json(path, payload)
