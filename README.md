# routing-aware-atos

This repo implements Routing Aware Activation Transport Operators and a small paper-facing experiment stack around them.

It supports:

- same-token baseline ATO training
- routing-aware operator training with `same_token`, `attention_top1`, `attention_topk`, and `attribution_topk`
- control routing policies: `previous_token`, `next_token`, `uniform_topk`, `random_topk`, and `shuffled_attention_topk`
- multi-source concatenated inputs for top-k routes via `input_mode: concat`
- feature-space evaluation
- causal restoration evaluation
- live model causal restoration utilities for logit / KL / next-token loss recovery
- transport taxonomy generation
- feature case study export
- full multi-policy experiment sweeps

The intended paper-scale model setup is:

- base model: **Gemma 2 2B**
- SAE family: **Gemma Scope**
- default residual SAE release: **`gemma-scope-2b-pt-res-canonical`**

## Paper-Scale Dataset And Protocol

The paper-scale workflow uses **`cerebras/SlimPajama-627B`**, streamed from Hugging Face. It does not use generated prompts or the mock cache.

The checked-in paper-scale configuration, `configs/real/gemma2_2b_slimpajama.yaml`, follows the paper protocol:

- 250,000 deterministically shuffled SlimPajama tokens for transport operators
- exact 60% train / 20% validation / 20% test token budgets
- a disjoint causal split containing 100 full sequences of 256 tokens from SlimPajama's test split
- 120,000 training tokens for Gemma Scope feature selection
- 819 selected features per target layer (the released selector's integer 5% cutoff)
- Gemma 2 2B post-layer residuals aligned to Gemma Scope layer indices
- float32 model computation and activation storage, matching the paper setup
- target layers 10 and 20 with the complete `k = 1..10` sweep
- same-token, routing-aware, random, shuffled-attention, attention-value proxy, and concatenated Top-K policies
- causal-prefix source constraints for every paper-scale routing policy, preventing future-token leakage
- live Gemma vocabulary-focus and candidate-feature causal-ablation scoring
- five-fold ridge grid search on training data, followed by untouched validation and test reporting

The source dataset is [Cerebras SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B), released under Apache 2.0. The paper reports the same dataset and token counts in Appendix B.

Reproducibility note: the paper names `cerebras/SlimPajama-627B`, while the authors' currently published code config points to `DKYoon/SlimPajama-6B`. This repository follows the paper text by default and pins the model and dataset revisions. The manifest also records the seed, shuffle buffer, config hash, software versions, collector source hash, and a SHA-256 digest for every activation shard. A strict released-code replication can change `dataset.name` to `DKYoon/SlimPajama-6B` and pin its commit before collection; it must use a separate `collection.output_dir` so the two corpora cannot be mixed accidentally.

### Registered Replication Differences

The checked-in paper-scale configuration produces actual model results, but it is not presented as bit-for-bit reproduction of the released repository:

- Feature scoring uses the released score weights, vocabulary projection, and live candidate ablations. Its bounded-memory coherence estimate counts active-token associations in 2,048 deterministic hash bins; the released selector keeps top-activation token dictionaries per chunk.
- The target-layer sweep covers `k = 1..10`; the transport-efficiency rank sweep uses the paper's displayed `k = 1, 3, 7, 10` offsets.
- The paper's all-target-layer heatmap is not part of the default run; the default evaluates the two preregistered SAE target layers, 10 and 20.
- `attention_value_proxy_topk` is explicitly a norm-weighted attention heuristic, not gradient attribution.
- Causal perplexity examples come from SlimPajama's disjoint test split after the first 10,000 examples, without shuffling. The released dataset config points to its training split.

These choices, immutable revisions, artifact hashes, and stage-specific implementation hashes are written into output artifacts. Use a separate output directory for any protocol variant.

## Install

```bash
uv sync
```

Install test dependencies:

```bash
uv sync --extra test
```

Optional paper-scale collection dependencies:

```bash
uv sync --extra real-model
```

For the full research environment:

```bash
uv sync --frozen --extra test --extra real-model --extra gemma-scope
```

Gemma 2 is gated. Accept its Hugging Face license and authenticate before collection:

```bash
hf auth login
```

Run tests:

```bash
uv run pytest -q
```

## Paper-Scale A100 Experiment

Run the complete resumable workflow with one command:

```bash
uv run python scripts/run_real_pipeline.py \
  --config configs/real/gemma2_2b_slimpajama.yaml \
  --stage all
```

The stages can also be run separately. This is useful on rented hardware because every expensive stage writes resumable artifacts:

```bash
uv run python scripts/run_real_pipeline.py --stage collect
uv run python scripts/run_real_pipeline.py --stage sae
uv run python scripts/run_real_pipeline.py --stage features
uv run python scripts/run_real_pipeline.py --stage experiments
uv run python scripts/run_real_pipeline.py --stage causal
uv run python scripts/run_real_pipeline.py --stage report
```

Activation collection resumes at verified shard boundaries after interruption. Feature, operator, and causal stages skip completed outputs unless `--force` is supplied.

Check a stage before spending GPU time:

```bash
uv run python scripts/preflight_real_experiment.py \
  --config configs/real/gemma2_2b_slimpajama.yaml \
  --stage collect
```

For the first A100 validation, run only one layer pair and the core baseline/comparison after collection, SAE export, and feature selection:

```bash
uv run python scripts/run_real_pipeline.py --stage experiments \
  --pair 9:10 \
  --policy same_token \
  --policy attention_topk
```

Paper-scale outputs are written under:

- `artifacts/gemma2_2b_slimpajama_250k/`: chunked residual, attention, split, and token cache
- `artifacts/sae/`: exported Gemma Scope encoder/decoder artifacts
- `artifacts/features/`: selected feature IDs and selection statistics
- `outputs/real/runs/`: trained operators and held-out metrics
- `outputs/real/live_causal/`: clean, zero-intervened, and ATO-restored perplexity results
- `outputs/real/report/`: validated CSV tables and predictive, efficiency, and causal plots

The paper-scale pipeline uses the model itself for causal logits and perplexity. It does not require the placeholder `gemma2_2b_readout.npz` used by the lightweight static demos.

## Core Data Model

The routed pipeline uses a cached sample format with:

- `tokens`: `list[str]` or `list[int]`
- `residuals`: `dict[int, np.ndarray]` with shape `[seq_len, d_model]`
- `attention_scores`: optional `dict[(source_layer, target_layer), np.ndarray]` with shape `[seq_len, seq_len]`
- `attribution_scores`: optional `dict[(source_layer, target_layer), np.ndarray]` with shape `[seq_len, seq_len]`

The matrix row for target position `i` contains source-token routing scores for that target.

## Quickstart

This section exercises individual components and mock-compatible configs. Use **Paper-Scale A100 Experiment** above for paper-scale results.

Build baseline pairs:

```bash
uv run python scripts/build_baseline_pairs.py --config configs/routing/same_token.yaml
```

Build routed pairs:

```bash
uv run python scripts/build_routed_pairs.py --config configs/routing/attention_topk.yaml
uv run python scripts/build_routed_pairs.py --config configs/routing/random_topk.yaml
uv run python scripts/build_routed_pairs.py --config configs/routing/shuffled_attention_topk.yaml
uv run python scripts/build_routed_pairs.py --config configs/routing/attention_topk_concat.yaml
```

Train a same-token baseline operator:

```bash
uv run python scripts/train_baseline_ato.py --config configs/experiment/train_baseline.yaml
```

Train a routed operator:

```bash
uv run python scripts/train_ra_atos.py --config configs/experiment/train_attention_topk.yaml
uv run python scripts/train_ra_atos.py --config configs/experiment/train_attention_topk_concat.yaml
```

Evaluate in feature space:

```bash
uv run python scripts/eval_feature_space.py --config configs/evaluation/attention_topk_feature_eval.yaml
```

Evaluate transport efficiency / CCA ceiling:

```bash
uv run python scripts/eval_transport_efficiency.py --config configs/evaluation/transport_efficiency.yaml
```

Run causal restoration:

```bash
uv run python scripts/run_causal_restore.py --config configs/evaluation/attention_topk_causal_restore.yaml
```

## Gemma 2 2B + Gemma Scope Setup

If you want to mirror the original paper setup more closely, use:

- cached activations collected from **Gemma 2 2B**
- residual-stream SAEs from **Gemma Scope**

The prompt-file collector command below is retained for small custom datasets. The actual SlimPajama workflow is configured and collected by `scripts/run_real_pipeline.py`.

Collect a custom prompt-file residual / attention cache:

```bash
uv run python scripts/collect_hf_activations.py --config configs/collection/hf_gemma2_2b.yaml
```

This repo expects SAE artifacts in `.npz` format. The exporter now stores decoder, encoder, biases, and JumpReLU threshold when available:

```bash
uv run python scripts/export_gemma_scope_decoder.py \
  --release gemma-scope-2b-pt-res-canonical \
  --sae-id layer_20/width_16k/canonical \
  --output artifacts/sae/gemma_scope_layer20_width16k.npz
```

The script also writes a JSON sidecar with the release and SAE metadata.

The paper-scale runner resolves target-layer SAE artifacts directly from the master configuration.

## Recommended Experiment Order

This repo does not use the original ATO repo entry points like `collect_activations.py`, `eval.py`, or `causal_perplexity_eval.py`. Use the scripts below instead.

### 1. Build training pairs

Baseline:

```bash
uv run python scripts/build_baseline_pairs.py --config configs/routing/same_token.yaml
```

Routed:

```bash
uv run python scripts/build_routed_pairs.py --config configs/routing/attention_top1.yaml
uv run python scripts/build_routed_pairs.py --config configs/routing/attention_topk.yaml
uv run python scripts/build_routed_pairs.py --config configs/routing/attribution_topk.yaml
uv run python scripts/build_routed_pairs.py --config configs/routing/previous_token.yaml
uv run python scripts/build_routed_pairs.py --config configs/routing/random_topk.yaml
uv run python scripts/build_routed_pairs.py --config configs/routing/shuffled_attention_topk.yaml
uv run python scripts/build_routed_pairs.py --config configs/routing/attention_topk_concat.yaml
```

### 2. Train operators

Baseline:

```bash
uv run python scripts/train_baseline_ato.py --config configs/experiment/train_baseline.yaml
```

Same-token through the routed stack:

```bash
uv run python scripts/train_ra_atos.py --config configs/experiment/train_same_token_routed.yaml
```

Attention top-1 / top-k / attribution top-k:

```bash
uv run python scripts/train_ra_atos.py --config configs/experiment/train_attention_top1.yaml
uv run python scripts/train_ra_atos.py --config configs/experiment/train_attention_topk.yaml
uv run python scripts/train_ra_atos.py --config configs/experiment/train_attribution_topk.yaml
uv run python scripts/train_ra_atos.py --config configs/experiment/train_previous_token.yaml
uv run python scripts/train_ra_atos.py --config configs/experiment/train_random_topk.yaml
uv run python scripts/train_ra_atos.py --config configs/experiment/train_shuffled_attention_topk.yaml
uv run python scripts/train_ra_atos.py --config configs/experiment/train_attention_topk_concat.yaml
```

### 3. Feature-space evaluation

```bash
uv run python scripts/eval_feature_space.py --config configs/evaluation/same_token_feature_eval.yaml
uv run python scripts/eval_feature_space.py --config configs/evaluation/attention_top1_feature_eval.yaml
uv run python scripts/eval_feature_space.py --config configs/evaluation/attention_topk_feature_eval.yaml
uv run python scripts/eval_feature_space.py --config configs/evaluation/attribution_topk_feature_eval.yaml
```

### 4. Transport efficiency / LTS estimate

```bash
uv run python scripts/eval_transport_efficiency.py --config configs/evaluation/transport_efficiency.yaml
```

This computes the CCA R2 ceiling, ATO R2 in whitened target space, efficiency ratio, and effective dimensionality estimate used for the linear transport subspace analysis.

### 5. Causal restoration evaluation

```bash
uv run python scripts/run_causal_restore.py --config configs/evaluation/same_token_causal_restore.yaml
uv run python scripts/run_causal_restore.py --config configs/evaluation/attention_top1_causal_restore.yaml
uv run python scripts/run_causal_restore.py --config configs/evaluation/attention_topk_causal_restore.yaml
uv run python scripts/run_causal_restore.py --config configs/evaluation/attribution_topk_causal_restore.yaml
```

### 6. Build taxonomy and case studies

Using the dedicated configs:

```bash
uv run python scripts/build_transport_taxonomy.py --config configs/evaluation/transport_taxonomy.yaml
uv run python scripts/export_feature_case_studies.py --config configs/evaluation/feature_case_studies.yaml
```

Or using the root eval config:

```bash
uv run python scripts/build_transport_taxonomy.py --config configs/eval.yaml
uv run python scripts/export_feature_case_studies.py --config configs/eval.yaml
```

## Full Sweep

The command in this section is the lightweight cached-config sweep. It assumes all paths already exist and is not the paper-scale data collector or live causal pipeline.

Run the full multi-policy workflow with:

```bash
uv run python scripts/run_full_routing_sweep.py --config configs/default.yaml
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
- The legacy root configs remain model-agnostic and require user-provided artifacts. The paper-scale pipeline above collects Gemma activations, exports Gemma Scope SAEs, and evaluates causal logits directly from Gemma; only the lightweight static demos use a readout export.
- The taxonomy builder accepts either:
  - explicit `runs:` payloads, or
  - `results_dir` plus `taxonomy_policies`
- The taxonomy builder supports both current output naming patterns:
  - `feature_metrics_<policy>.json`
  - `<policy>_feature_eval.json`
- If you want to evaluate directly from cached samples instead of saved pair files, use the routed-capable loaders and configs wired into `scripts/eval_feature_space.py`, `scripts/run_causal_restore.py`, and `scripts/train_transport_operators.py`.
