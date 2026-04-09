# routing-aware-atos

This repo implements Routing Aware Activation Transport Operators and a small paper-facing experiment stack around them.

It supports:

- same-token baseline ATO training
- routing-aware operator training with `same_token`, `attention_top1`, `attention_topk`, and `attribution_topk`
- feature-space evaluation
- causal restoration evaluation
- transport taxonomy generation
- feature case study export
- full multi-policy experiment sweeps

## Install

```bash
python -m venv .routing
.routing\Scripts\activate
pip install -e .
```

If you already have the local repo environment, use that instead.

## Core Data Model

The routed pipeline uses a cached sample format with:

- `tokens`: `list[str]` or `list[int]`
- `residuals`: `dict[int, np.ndarray]` with shape `[seq_len, d_model]`
- `attention_scores`: optional `dict[(source_layer, target_layer), np.ndarray]` with shape `[seq_len, seq_len]`
- `attribution_scores`: optional `dict[(source_layer, target_layer), np.ndarray]` with shape `[seq_len, seq_len]`

The matrix row for target position `i` contains source-token routing scores for that target.

## Quickstart

Build baseline pairs:

```bash
python scripts/build_baseline_pairs.py --config configs/routing/same_token.yaml
```

Build routed pairs:

```bash
python scripts/build_routed_pairs.py --config configs/routing/attention_topk.yaml
```

Train a same-token baseline operator:

```bash
python scripts/train_baseline_ato.py --config configs/experiment/train_baseline.yaml
```

Train a routed operator:

```bash
python scripts/train_ra_atos.py --config configs/experiment/train_attention_topk.yaml
```

Evaluate in feature space:

```bash
python scripts/eval_feature_space.py --config configs/evaluation/attention_topk_feature_eval.yaml
```

Run causal restoration:

```bash
python scripts/run_causal_restore.py --config configs/evaluation/attention_topk_causal_restore.yaml
```

## Recommended Experiment Order

This repo does not use the original ATO repo entry points like `collect_activations.py`, `eval.py`, or `causal_perplexity_eval.py`. Use the scripts below instead.

### 1. Build training pairs

Baseline:

```bash
python scripts/build_baseline_pairs.py --config configs/routing/same_token.yaml
```

Routed:

```bash
python scripts/build_routed_pairs.py --config configs/routing/attention_top1.yaml
python scripts/build_routed_pairs.py --config configs/routing/attention_topk.yaml
python scripts/build_routed_pairs.py --config configs/routing/attribution_topk.yaml
```

### 2. Train operators

Baseline:

```bash
python scripts/train_baseline_ato.py --config configs/experiment/train_baseline.yaml
```

Same-token through the routed stack:

```bash
python scripts/train_ra_atos.py --config configs/experiment/train_same_token_routed.yaml
```

Attention top-1 / top-k / attribution top-k:

```bash
python scripts/train_ra_atos.py --config configs/experiment/train_attention_top1.yaml
python scripts/train_ra_atos.py --config configs/experiment/train_attention_topk.yaml
python scripts/train_ra_atos.py --config configs/experiment/train_attribution_topk.yaml
```

### 3. Feature-space evaluation

```bash
python scripts/eval_feature_space.py --config configs/evaluation/same_token_feature_eval.yaml
python scripts/eval_feature_space.py --config configs/evaluation/attention_top1_feature_eval.yaml
python scripts/eval_feature_space.py --config configs/evaluation/attention_topk_feature_eval.yaml
python scripts/eval_feature_space.py --config configs/evaluation/attribution_topk_feature_eval.yaml
```

### 4. Causal restoration evaluation

```bash
python scripts/run_causal_restore.py --config configs/evaluation/same_token_causal_restore.yaml
python scripts/run_causal_restore.py --config configs/evaluation/attention_top1_causal_restore.yaml
python scripts/run_causal_restore.py --config configs/evaluation/attention_topk_causal_restore.yaml
python scripts/run_causal_restore.py --config configs/evaluation/attribution_topk_causal_restore.yaml
```

### 5. Build taxonomy and case studies

Using the dedicated configs:

```bash
python scripts/build_transport_taxonomy.py --config configs/evaluation/transport_taxonomy.yaml
python scripts/export_feature_case_studies.py --config configs/evaluation/feature_case_studies.yaml
```

Or using the root eval config:

```bash
python scripts/build_transport_taxonomy.py --config configs/eval.yaml
python scripts/export_feature_case_studies.py --config configs/eval.yaml
```

## Full Sweep

Run the full multi-policy workflow with:

```bash
python scripts/run_full_routing_sweep.py --config configs/default.yaml
```

That script resolves per-policy configs for:

- training
- feature evaluation
- causal restoration
- taxonomy build
- case study export

## Important Config Files

Root configs:

- `configs/default.yaml`
- `configs/eval.yaml`
- `configs/causal_eval.yaml`

Experiment configs:

- `configs/experiment/train_baseline.yaml`
- `configs/experiment/train_same_token_routed.yaml`
- `configs/experiment/train_attention_top1.yaml`
- `configs/experiment/train_attention_topk.yaml`
- `configs/experiment/train_attribution_topk.yaml`

Evaluation configs:

- `configs/evaluation/same_token_feature_eval.yaml`
- `configs/evaluation/attention_top1_feature_eval.yaml`
- `configs/evaluation/attention_topk_feature_eval.yaml`
- `configs/evaluation/attribution_topk_feature_eval.yaml`
- `configs/evaluation/same_token_causal_restore.yaml`
- `configs/evaluation/attention_top1_causal_restore.yaml`
- `configs/evaluation/attention_topk_causal_restore.yaml`
- `configs/evaluation/attribution_topk_causal_restore.yaml`
- `configs/evaluation/transport_taxonomy.yaml`
- `configs/evaluation/feature_case_studies.yaml`

## Notes

- The root `eval.yaml` and `causal_eval.yaml` provide shared routing/taxonomy defaults.
- The taxonomy builder accepts either:
  - explicit `runs:` payloads, or
  - `results_dir` plus `taxonomy_policies`
- The taxonomy builder supports both current output naming patterns:
  - `feature_metrics_<policy>.json`
  - `<policy>_feature_eval.json`
- If you want to evaluate directly from cached samples instead of saved pair files, use the routed-capable loaders and configs wired into `scripts/eval_feature_space.py`, `scripts/run_causal_restore.py`, and `scripts/train_transport_operators.py`.
