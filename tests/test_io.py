from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from routing_aware_atos.utils import io
from routing_aware_atos.utils.io import load_json, load_npz, save_json, save_npz


def test_atomic_json_failure_preserves_previous_artifact(tmp_path: Path):
    path = tmp_path / "artifact.json"
    save_json(path, {"version": 1})

    with pytest.raises(ValueError):
        save_json(path, {"invalid": float("nan")})

    assert load_json(path) == {"version": 1}
    assert list(tmp_path.glob(".artifact.json.*")) == []


def test_atomic_npz_failure_preserves_previous_artifact(tmp_path: Path, monkeypatch):
    path = tmp_path / "artifact.npz"
    save_npz(path, values=np.asarray([1, 2, 3]))

    def fail_save(destination, **arrays):
        Path(destination).write_bytes(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(io.np, "savez", fail_save)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        save_npz(path, values=np.asarray([4, 5, 6]))

    np.testing.assert_array_equal(load_npz(path)["values"], [1, 2, 3])
    assert list(tmp_path.glob(".artifact.npz.*")) == []
