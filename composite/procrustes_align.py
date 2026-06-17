#!/usr/bin/env python3
"""
procrustes_align.py — Compute bridge matrices between model representation spaces.

Uses the Procrustes problem (analytical SVD solution) to find the optimal linear
transformation that maps hidden states from one model's representation space to
another. NO gradient training — pure linear algebra.

For dimension expansion (smaller → larger), uses least-squares projection.
For same-dimension, creates identity bridges.
"""

import os
import sys
import json
import time
import logging
import gc
import itertools

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODELS, HIDDEN_DIMS, LAYER_RANGES,
    BRIDGE_DIR, LAYER_CACHE_DIR,
    PROCRUSTES_CALIBRATION_SAMPLES,
    CALIBRATION_DATASET,
    DTYPE, SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

torch.manual_seed(SEED)


def collect_hidden_states(hf_id: str, layer_idx: int, prompts: list,
                          tokenizer, device: str, max_samples: int = 256):
    """
    Load a model, run prompts through it, and collect hidden states
    at a specific layer index. Returns tensor of shape (N, hidden_dim).
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    load_kwargs = {
        "torch_dtype": getattr(torch, DTYPE),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "output_hidden_states": True,
    }
    if "cuda" in device:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    model.eval()

    all_states = []
    batch_size = 4

    for start in range(0, min(len(prompts), max_samples), batch_size):
        end = min(start + batch_size, len(prompts), max_samples)
        batch = prompts[start:end]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        with torch.no_grad(), autocast(dtype=getattr(torch, DTYPE)):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # Take hidden states at the specified layer
        # outputs.hidden_states[layer_idx + 1] = output of layer layer_idx
        hs_idx = min(layer_idx + 1, len(outputs.hidden_states) - 1)
        h = outputs.hidden_states[hs_idx]  # (batch, seq, hidden_dim)

        # Use mean-pooled representation (across non-padding tokens)
        mask_expanded = attn_mask.unsqueeze(-1).float()
        h_pooled = (h.float() * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        all_states.append(h_pooled.cpu())

        del outputs, input_ids, attn_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.cat(all_states, dim=0)  # (N, hidden_dim)


def compute_procrustes_bridge(h_source: torch.Tensor, h_target: torch.Tensor):
    """
    Compute the optimal bridge matrix from source space to target space.

    If dimensions match: orthogonal Procrustes (W = U @ Vt from SVD of cross-covariance).
    If dimensions differ: least-squares solution (W = lstsq(h_source, h_target)).

    Returns: bridge matrix W (dim_target x dim_source), alignment quality (cosine sim).
    """
    dim_source = h_source.shape[1]
    dim_target = h_target.shape[1]
    N = min(h_source.shape[0], h_target.shape[0])

    # Ensure same number of samples
    h_s = h_source[:N].float()
    h_t = h_target[:N].float()

    # Center the data for better alignment
    h_s_mean = h_s.mean(dim=0, keepdim=True)
    h_t_mean = h_t.mean(dim=0, keepdim=True)
    h_s_centered = h_s - h_s_mean
    h_t_centered = h_t - h_t_mean

    if dim_source == dim_target:
        # Orthogonal Procrustes: minimize ||W @ h_s - h_t||
        # Solution: W = U @ Vt from SVD of h_t.T @ h_s
        C = h_t_centered.T @ h_s_centered  # (dim x dim)
        U, S, Vt = torch.linalg.svd(C, full_matrices=False)
        W = U @ Vt  # orthogonal bridge
        log.info(f"  Procrustes (orthogonal): {dim_source}→{dim_target}, "
                 f"singular values range: [{S.min():.2f}, {S.max():.2f}]")
    else:
        # Non-square: use least-squares
        # We want W such that h_s @ W.T ≈ h_t
        # i.e., W.T = lstsq(h_s, h_t) → W = solution.T
        result = torch.linalg.lstsq(h_s_centered, h_t_centered)
        W = result.solution.T  # (dim_target x dim_source)
        log.info(f"  Least-squares bridge: {dim_source}→{dim_target}")

    # Compute alignment quality: cosine similarity of mapped source vs target
    h_mapped = (h_s_centered @ W.T)  # (N, dim_target)
    cos_sim = F.cosine_similarity(h_mapped, h_t_centered, dim=-1).mean().item()

    # Store means for runtime centering (optional refinement)
    bridge_data = {
        "weight": W,
        "source_mean": h_s_mean.squeeze(0),
        "target_mean": h_t_mean.squeeze(0),
        "alignment_quality": cos_sim,
        "dim_in": dim_source,
        "dim_out": dim_target,
    }

    return bridge_data


def load_calibration_prompts():
    """Load GSM8K calibration prompts."""
    from datasets import load_dataset

    ds = load_dataset(CALIBRATION_DATASET, "main", split="train")
    ds = ds.select(range(min(PROCRUSTES_CALIBRATION_SAMPLES, len(ds))))
    prompts = [f"Question: {row['question']}\nAnswer:" for row in ds]
    log.info(f"Loaded {len(prompts)} calibration prompts.")
    return prompts


def main():
    log.info("=" * 70)
    log.info("COMPOSITE MODEL — PHASE 3: Procrustes Alignment")
    log.info("=" * 70)

    from transformers import AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Load calibration prompts
    prompts = load_calibration_prompts()

    # Identify unique dimension pairs that need bridges
    unique_dims = set(HIDDEN_DIMS.values())
    model_names = list(MODELS.keys())

    # We need bridges between every pair of models that could be adjacent
    # in the composite layer sequence. In practice, we compute ALL pairwise bridges.
    log.info(f"Unique hidden dimensions: {sorted(unique_dims)}")
    log.info(f"Models: {model_names}")

    # Collect hidden states from each model at its representative middle layer
    model_hidden_states = {}

    for i, name in enumerate(model_names):
        hf_id = MODELS[name]
        layer_start, layer_end = LAYER_RANGES[name]
        # Use the middle layer as the representative
        rep_layer = (layer_start + layer_end) // 2

        gpu_id = i % num_gpus
        dev = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        log.info(f"[{name}] Collecting hidden states at layer {rep_layer} on GPU {gpu_id} ...")

        # Load tokenizer for this model
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        h = collect_hidden_states(hf_id, rep_layer, prompts, tokenizer, dev,
                                  max_samples=PROCRUSTES_CALIBRATION_SAMPLES)
        model_hidden_states[name] = h
        log.info(f"[{name}] Hidden states collected: shape {h.shape}")

        del tokenizer
        gc.collect()

    # Compute bridges for all pairs
    log.info("\nComputing bridge matrices for all model pairs ...")
    results_summary = []

    for (name_a, name_b) in itertools.combinations(model_names, 2):
        dim_a = HIDDEN_DIMS[name_a]
        dim_b = HIDDEN_DIMS[name_b]

        h_a = model_hidden_states[name_a]
        h_b = model_hidden_states[name_b]

        # Compute A→B bridge
        log.info(f"\n[{name_a}({dim_a}) → {name_b}({dim_b})]")
        bridge_ab = compute_procrustes_bridge(h_a, h_b)
        save_path_ab = os.path.join(
            BRIDGE_DIR,
            f"bridge_{name_a}_{dim_a}_to_{name_b}_{dim_b}.pt"
        )
        torch.save(bridge_ab, save_path_ab)
        log.info(f"  Saved: {save_path_ab}")
        log.info(f"  Alignment quality (cosine sim): {bridge_ab['alignment_quality']:.4f}")

        # Compute B→A bridge
        log.info(f"[{name_b}({dim_b}) → {name_a}({dim_a})]")
        bridge_ba = compute_procrustes_bridge(h_b, h_a)
        save_path_ba = os.path.join(
            BRIDGE_DIR,
            f"bridge_{name_b}_{dim_b}_to_{name_a}_{dim_a}.pt"
        )
        torch.save(bridge_ba, save_path_ba)
        log.info(f"  Saved: {save_path_ba}")
        log.info(f"  Alignment quality (cosine sim): {bridge_ba['alignment_quality']:.4f}")

        results_summary.append({
            "source": name_a, "target": name_b,
            "dim_source": dim_a, "dim_target": dim_b,
            "quality_forward": bridge_ab["alignment_quality"],
            "quality_reverse": bridge_ba["alignment_quality"],
        })

    # Also create identity bridges for same-dim models
    for name in model_names:
        dim = HIDDEN_DIMS[name]
        bridge_data = {
            "weight": torch.eye(dim),
            "source_mean": torch.zeros(dim),
            "target_mean": torch.zeros(dim),
            "alignment_quality": 1.0,
            "dim_in": dim,
            "dim_out": dim,
        }
        path = os.path.join(BRIDGE_DIR, f"bridge_{name}_{dim}_to_{name}_{dim}.pt")
        torch.save(bridge_data, path)

    # Save summary
    summary_path = os.path.join(BRIDGE_DIR, "alignment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    # Print results table
    log.info("\n" + "=" * 70)
    log.info("ALIGNMENT QUALITY SUMMARY:")
    log.info(f"  {'Source':>15} → {'Target':>15}  {'Dims':>12}  {'Quality':>8}")
    log.info(f"  {'─' * 60}")
    for r in results_summary:
        status = "✅" if r["quality_forward"] > 0.60 else "⚠️"
        log.info(f"  {r['source']:>15} → {r['target']:>15}  "
                 f"{r['dim_source']:>5}→{r['dim_target']:<5}  "
                 f"{r['quality_forward']:>8.4f}  {status}")
        log.info(f"  {r['target']:>15} → {r['source']:>15}  "
                 f"{r['dim_target']:>5}→{r['dim_source']:<5}  "
                 f"{r['quality_reverse']:>8.4f}  {status}")

    log.info(f"\nBridge files saved to: {BRIDGE_DIR}")
    log.info(f"Total bridges computed: {len(results_summary) * 2 + len(model_names)}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
