from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

import numpy as np

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
            "`pip install -e .[gemma-scope]` or `pip install sae-lens`."
        ) from exc

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=args.release,
        sae_id=args.sae_id,
        device=args.device,
    )

    decoder = _extract_numpy_array(sae, "W_dec")
    if decoder is None:
        raise RuntimeError("Loaded SAE does not expose `W_dec`.")

    arrays = {"decoder": decoder}
    encoder = _extract_numpy_array(sae, "W_enc")
    if encoder is not None:
        arrays["encoder"] = encoder
    b_dec = _extract_numpy_array(sae, "b_dec")
    if b_dec is not None:
        arrays["b_dec"] = b_dec

    save_npz(args.output, **arrays)

    metadata = {
        "model_family": "gemma-2",
        "model_name": "gemma-2-2b",
        "sae_family": "gemma-scope",
        "release": args.release,
        "sae_id": args.sae_id,
        "device": args.device,
        "decoder_shape": [int(x) for x in decoder.shape],
        "cfg_dict": cfg_dict,
        "sparsity": None if sparsity is None else float(sparsity),
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
