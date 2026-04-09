from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.utils.io import save_cached_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Create mock cached samples for demos and tests")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=4)
    args = parser.parse_args()

    samples = make_mock_samples(args.num_samples, args.seq_len, args.d_model)
    save_cached_samples(args.output, samples)
    print(f"Saved mock samples to {args.output}")


if __name__ == "__main__":
    main()
