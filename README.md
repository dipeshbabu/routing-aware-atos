# routing-aware-atos

This artifact implements Routing Aware Activation Transport Operators:

It also includes a minimal baseline reproduction path for same-token ATO pair building so you can compare against the original Activation Transport Operators pipeline.

## What is included

- A small routing policy framework
- Same-token, attention top-1, attention top-k, and attribution top-k policies
- A routed dataset builder that turns cached activations and routing scores into training pairs `(X, Y)`
- A baseline builder for standard same-token ATO pairs
- Simple config-driven scripts for pair construction
- Unit tests for routing and dataset behavior

## Install

```bash
pip install -e .
```

## Example: baseline same-token pairs

```bash
python scripts/build_baseline_pairs.py --config configs/routing/same_token.yaml
```

## Example: routed pairs

```bash
python scripts/build_routed_pairs.py --config configs/routing/attention_topk.yaml
```

## Expected cache format

The scripts assume a cached sample dictionary with keys:

- `tokens`: list[str] or list[int]
- `residuals`: dict[int, np.ndarray] with shape `[seq_len, d_model]`
- `attention_scores`: optional dict[tuple[int, int], np.ndarray] with shape `[seq_len, seq_len]`
- `attribution_scores`: optional dict[tuple[int, int], np.ndarray] with shape `[seq_len, seq_len]`

The `(source_layer, target_layer)` key indexes a matrix where row `i` contains source scores for target position `i`.
