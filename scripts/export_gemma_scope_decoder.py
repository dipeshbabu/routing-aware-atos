from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
from importlib.metadata import version

import numpy as np

from routing_aware_atos.provenance import sha256_file
from routing_aware_atos.utils.io import save_json, save_npz


def _extract_numpy_array(obj, attr_name: str) -> np.ndarray | None:
    value = getattr(obj, attr_name, None)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Gemma Scope SAE decoder into this repo's decoder.npz format."
    )
    parser.add_argument(
        "--release",
        type=str,
        default="gemma-scope-2b-pt-res-canonical",
        help="Gemma Scope release name, e.g. gemma-scope-2b-pt-res-canonical",
    )
    parser.add_argument(
        "--sae-id",
        type=str,
        required=True,
        help="SAE id inside the release, e.g. layer_20/width_16k/canonical",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz path that will contain a `decoder` array.",
    )
    parser.add_argument(
        "--metadata-output",
        type=str,
        help="Optional JSON sidecar path for release / SAE metadata.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device passed to SAE.from_pretrained.",
    )
    args = parser.parse_args()

    try:
        from sae_lens import SAE
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "Missing optional dependency `sae-lens`. Install it with "
            "`uv sync --extra gemma-scope`."
        ) from exc

    if hasattr(SAE, "from_pretrained_with_cfg_and_sparsity"):
        sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release=args.release,
            sae_id=args.sae_id,
            device=args.device,
            dtype="float32",
        )
    else:  # pragma: no cover - compatibility with older SAE Lens releases
        loaded = SAE.from_pretrained(
            release=args.release,
            sae_id=args.sae_id,
            device=args.device,
        )
        if not isinstance(loaded, tuple) or len(loaded) != 3:
            raise RuntimeError("Unsupported SAE Lens pretrained-loading API")
        sae, cfg_dict, sparsity = loaded

    decoder = _extract_numpy_array(sae, "W_dec")
    if decoder is None:
        raise RuntimeError("Loaded SAE does not expose `W_dec`.")

    arrays = {
        "decoder": decoder,
        "architecture": np.asarray(str(cfg_dict.get("architecture") or "unknown")),
        "activation_fn": np.asarray(str(cfg_dict.get("activation_fn") or "relu")),
        "apply_b_dec_to_input": np.asarray(
            bool(cfg_dict.get("apply_b_dec_to_input", True)),
            dtype=np.uint8,
        ),
        "normalize_activations": np.asarray(
            str(cfg_dict.get("normalize_activations") or "none")
        ),
    }
    if cfg_dict.get("activation_normalization_factor") is not None:
        arrays["activation_normalization_factor"] = np.asarray(
            float(cfg_dict["activation_normalization_factor"]),
            dtype=np.float32,
        )
    encoder = _extract_numpy_array(sae, "W_enc")
    if encoder is not None:
        arrays["encoder"] = encoder
    b_dec = _extract_numpy_array(sae, "b_dec")
    if b_dec is not None:
        arrays["b_dec"] = b_dec
    b_enc = _extract_numpy_array(sae, "b_enc")
    if b_enc is not None:
        arrays["b_enc"] = b_enc
    threshold = _extract_numpy_array(sae, "threshold")
    if threshold is not None:
        arrays["threshold"] = threshold
    scaling_factor = _extract_numpy_array(sae, "scaling_factor")
    if scaling_factor is not None:
        arrays["scaling_factor"] = scaling_factor

    save_npz(args.output, **arrays)
    artifact_sha256 = sha256_file(args.output)

    sparsity_metadata = None
    if sparsity is not None:
        sparsity_value = sparsity.detach() if hasattr(sparsity, "detach") else sparsity
        sparsity_value = (
            sparsity_value.cpu() if hasattr(sparsity_value, "cpu") else sparsity_value
        )
        sparsity_array = np.asarray(sparsity_value, dtype=np.float32)
        sparsity_metadata = {
            "shape": [int(value) for value in sparsity_array.shape],
            "mean": float(np.mean(sparsity_array)),
        }

    metadata = {
        "model_family": "gemma-2",
        "model_name": "gemma-2-2b",
        "sae_family": "gemma-scope",
        "release": args.release,
        "sae_id": args.sae_id,
        "device": args.device,
        "sae_lens_version": version("sae-lens"),
        "artifact_sha256": artifact_sha256,
        "decoder_shape": [int(x) for x in decoder.shape],
        "encoder_shape": None if encoder is None else [int(x) for x in encoder.shape],
        "has_threshold": threshold is not None,
        "architecture": str(cfg_dict.get("architecture") or "unknown"),
        "activation_fn": str(cfg_dict.get("activation_fn") or "relu"),
        "apply_b_dec_to_input": bool(cfg_dict.get("apply_b_dec_to_input", True)),
        "normalize_activations": str(cfg_dict.get("normalize_activations") or "none"),
        "cfg_dict": cfg_dict,
        "sparsity": sparsity_metadata,
    }

    metadata_output = (
        Path(args.metadata_output)
        if args.metadata_output
        else Path(args.output).with_suffix(".json")
    )
    save_json(metadata_output, metadata)

    print(f"Saved Gemma Scope decoder to {args.output}")
    print(f"Saved metadata to {metadata_output}")


if __name__ == "__main__":
    main()
