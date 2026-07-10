from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json

import numpy as np
import torch

from routing_aware_atos.routed_types import CachedSample
from routing_aware_atos.utils.io import load_yaml, save_cached_samples


def _load_prompts(path: str | Path) -> list[str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(x) for x in payload]
        if isinstance(payload, dict) and "prompts" in payload:
            return [str(x) for x in payload["prompts"]]
        raise ValueError("JSON prompts file must be a list or contain a 'prompts' list")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _attention_index(target_layer: int, num_attention_layers: int) -> int:
    if 0 <= target_layer < num_attention_layers:
        return target_layer
    shifted = target_layer - 1
    if 0 <= shifted < num_attention_layers:
        return shifted
    raise IndexError(f"Cannot map target_layer={target_layer} to {num_attention_layers} attention tensors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect residual and attention caches from a Hugging Face causal LM")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install the real-model extra or transformers to use this script") from exc

    model_name = cfg["model_name"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = cfg.get("dtype", "float32")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {"torch_dtype": dtype}
    if cfg.get("attn_implementation") is not None:
        model_kwargs["attn_implementation"] = cfg["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()

    layer_indices = [int(x) for x in cfg["layer_indices"]]
    attention_pairs = [tuple(int(v) for v in pair) for pair in cfg.get("attention_layer_pairs", [])]
    prompts = _load_prompts(cfg["prompts_path"])
    max_length = cfg.get("max_length", 256)

    samples: list[CachedSample] = []
    for sample_idx, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=bool(attention_pairs),
                use_cache=False,
            )

        residuals = {}
        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= len(outputs.hidden_states):
                raise IndexError(f"layer_idx={layer_idx} out of bounds for {len(outputs.hidden_states)} hidden states")
            residuals[layer_idx] = outputs.hidden_states[layer_idx][0].detach().float().cpu().numpy().astype(np.float32)

        attention_scores = None
        if attention_pairs:
            attention_scores = {}
            if outputs.attentions is None:
                raise RuntimeError("Model did not return attentions")
            for source_layer, target_layer in attention_pairs:
                attn_idx = _attention_index(target_layer, len(outputs.attentions))
                pooled = outputs.attentions[attn_idx][0].mean(dim=0)
                attention_scores[(source_layer, target_layer)] = pooled.detach().float().cpu().numpy().astype(np.float32)

        samples.append(
            CachedSample(
                tokens=input_ids[0].detach().cpu().tolist(),
                residuals=residuals,
                attention_scores=attention_scores,
                attribution_scores=None,
                metadata={
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "model_name": model_name,
                },
            )
        )

    save_cached_samples(cfg["output_path"], samples)
    print(f"Saved {len(samples)} cached samples -> {cfg['output_path']}")


if __name__ == "__main__":
    main()
