#!/usr/bin/env python3
"""
activation_fingerprint.py — Score each frontier layer for reasoning signal strength.

Runs calibration prompts through each full model, records per-layer metrics:
  - activation_change: how much the layer transforms its input (L2 norm ratio)
  - reasoning_gradient_signal: gradient magnitude at the layer wrt the correct answer
  - reasoning_score: weighted combination → used to rank the best layers

Output: per-model fingerprint JSON files + ranked layer lists.
"""

import os
import sys
import json
import time
import logging
import gc

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODELS, HIDDEN_DIMS, NUM_LAYERS, LAYER_RANGES,
    LAYER_CACHE_DIR, FINGERPRINT_DIR,
    CALIBRATION_DATASET, MAX_CALIBRATION_ROWS,
    DTYPE, SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

torch.manual_seed(SEED)


def load_calibration_data(tokenizer, max_rows: int = None):
    """Load GSM8K calibration prompts and tokenize them."""
    from datasets import load_dataset

    max_rows = max_rows or MAX_CALIBRATION_ROWS
    log.info(f"Loading {CALIBRATION_DATASET} calibration data ({max_rows} rows) ...")

    ds = load_dataset(CALIBRATION_DATASET, "main", split="train")
    ds = ds.select(range(min(max_rows, len(ds))))

    prompts = []
    answers = []
    for row in ds:
        q = row["question"]
        a = row["answer"]
        prompt = f"Question: {q}\nAnswer: {a}"
        prompts.append(prompt)
        answers.append(a)

    # Tokenize all prompts
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    log.info(f"Loaded {len(prompts)} calibration prompts.")
    return encodings, prompts, answers


def fingerprint_model(name: str, hf_id: str, gpu_id: int):
    """Run calibration data through a model and score each layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    layer_start, layer_end = LAYER_RANGES[name]
    out_path = os.path.join(FINGERPRINT_DIR, f"{name}_fingerprint.json")
    top_path = os.path.join(FINGERPRINT_DIR, f"{name}_top_layers.json")

    if os.path.exists(out_path) and os.path.exists(top_path):
        log.info(f"[{name}] Fingerprints already computed. Skipping.")
        return

    log.info(f"[{name}] Loading model for fingerprinting on GPU {gpu_id} ...")
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": getattr(torch, DTYPE),
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "output_hidden_states": True,
        }
        if torch.cuda.is_available():
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = {"": gpu_id}
        else:
            load_kwargs["device_map"] = "cpu"

        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        model.eval()

        # Load calibration data
        encodings, prompts, answers = load_calibration_data(tokenizer, max_rows=200)

        fingerprint = {}
        batch_size = 4  # small batches to avoid OOM

        log.info(f"[{name}] Running fingerprinting on layers {layer_start}-{layer_end - 1} ...")

        # Process in batches
        all_hidden_states = []
        all_logits = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_ids = encodings["input_ids"][batch_start:batch_end].to(device)
            batch_mask = encodings["attention_mask"][batch_start:batch_end].to(device)

            with torch.no_grad(), autocast(dtype=getattr(torch, DTYPE)):
                outputs = model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    output_hidden_states=True,
                )

            # outputs.hidden_states is tuple of (n_layers + 1) tensors
            # each tensor shape: (batch, seq_len, hidden_dim)
            # Collect per-layer norms for this batch
            hs = outputs.hidden_states  # tuple of tensors
            for layer_idx in range(layer_start, min(layer_end, len(hs) - 1)):
                h_in = hs[layer_idx]      # input to this layer
                h_out = hs[layer_idx + 1]  # output of this layer

                # Metric 1: Activation change (how much the layer transforms)
                delta = (h_out - h_in).float()
                delta_norm = delta.norm(dim=-1).mean().item()
                input_norm = h_in.float().norm(dim=-1).mean().item() + 1e-8
                activation_change = delta_norm / input_norm

                # Metric 2: Hidden state variance (diversity of representations)
                h_var = h_out.float().var(dim=-1).mean().item()

                if layer_idx not in fingerprint:
                    fingerprint[layer_idx] = {
                        "activation_change_sum": 0.0,
                        "hidden_state_variance_sum": 0.0,
                        "count": 0,
                    }
                fingerprint[layer_idx]["activation_change_sum"] += activation_change
                fingerprint[layer_idx]["hidden_state_variance_sum"] += h_var
                fingerprint[layer_idx]["count"] += 1

            # Free batch memory
            del outputs, batch_ids, batch_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Now compute gradient signal for each layer (more expensive)
        log.info(f"[{name}] Computing gradient reasoning signal ...")

        # Use a smaller subset for gradient computation (expensive)
        grad_samples = min(50, len(prompts))
        for layer_idx in range(layer_start, min(layer_end, NUM_LAYERS[name])):
            grad_signal_total = 0.0
            grad_count = 0

            for i in range(0, grad_samples, 2):
                batch_ids = encodings["input_ids"][i:i+2].to(device)
                batch_mask = encodings["attention_mask"][i:i+2].to(device)

                # Enable gradient computation for this specific forward pass
                model.zero_grad()

                # Hook to capture gradient at this layer
                grad_norms = []

                def hook_fn(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        grad_norms.append(grad_output[0].float().norm().item())

                # Get the layer
                layers_list = model.model.layers if hasattr(model, "model") else model.transformer.h
                if layer_idx < len(layers_list):
                    handle = layers_list[layer_idx].register_full_backward_hook(hook_fn)

                    try:
                        with autocast(dtype=getattr(torch, DTYPE)):
                            outputs = model(
                                input_ids=batch_ids,
                                attention_mask=batch_mask,
                                labels=batch_ids,  # compute loss
                            )
                        loss = outputs.loss
                        if loss is not None:
                            loss.backward()
                            if grad_norms:
                                grad_signal_total += sum(grad_norms) / len(grad_norms)
                                grad_count += 1
                    except Exception as e:
                        log.warning(f"  [{name}] Gradient computation failed at layer {layer_idx}: {e}")
                    finally:
                        handle.remove()
                        model.zero_grad()
                        del outputs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if layer_idx in fingerprint:
                fingerprint[layer_idx]["reasoning_gradient_signal"] = (
                    grad_signal_total / max(grad_count, 1)
                )

        # Normalize and compute final scores
        results = {}
        max_act = max((v["activation_change_sum"] / v["count"]
                       for v in fingerprint.values()), default=1.0)
        max_grad = max((v.get("reasoning_gradient_signal", 0.0)
                        for v in fingerprint.values()), default=1.0)

        for layer_idx, data in fingerprint.items():
            count = data["count"]
            act_change = data["activation_change_sum"] / count
            h_var = data["hidden_state_variance_sum"] / count
            grad_sig = data.get("reasoning_gradient_signal", 0.0)

            # Normalize to [0, 1]
            norm_act = act_change / max(max_act, 1e-8)
            norm_grad = grad_sig / max(max_grad, 1e-8)

            # Weighted reasoning score
            reasoning_score = 0.4 * norm_act + 0.6 * norm_grad

            results[str(layer_idx)] = {
                "activation_change": round(act_change, 6),
                "hidden_state_variance": round(h_var, 6),
                "reasoning_gradient_signal": round(grad_sig, 6),
                "reasoning_score": round(reasoning_score, 6),
            }

        # Save fingerprint
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"[{name}] Fingerprint saved to {out_path}")

        # Save ranked list
        ranked = sorted(
            results.items(),
            key=lambda x: x[1]["reasoning_score"],
            reverse=True,
        )
        top_layers = [{"layer_idx": int(k), **v} for k, v in ranked]
        with open(top_path, "w") as f:
            json.dump(top_layers, f, indent=2)
        log.info(f"[{name}] Top layers saved to {top_path}")

        # Print summary table
        log.info(f"\n[{name}] TOP 10 REASONING LAYERS:")
        log.info(f"  {'Layer':>6}  {'ActChange':>10}  {'GradSig':>10}  {'Score':>8}")
        log.info(f"  {'─' * 40}")
        for entry in top_layers[:10]:
            log.info(f"  {entry['layer_idx']:>6}  "
                     f"{entry['activation_change']:>10.4f}  "
                     f"{entry['reasoning_gradient_signal']:>10.4f}  "
                     f"{entry['reasoning_score']:>8.4f}")

    except Exception as e:
        log.error(f"[{name}] Fingerprinting FAILED: {e}", exc_info=True)
    finally:
        try:
            del model, tokenizer
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log.info(f"[{name}] Fingerprinting complete in {elapsed:.0f}s")


def main():
    log.info("=" * 70)
    log.info("COMPOSITE MODEL — PHASE 2: Activation Fingerprinting")
    log.info("=" * 70)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    log.info(f"Available GPUs: {num_gpus}")

    for i, (name, hf_id) in enumerate(MODELS.items()):
        gpu_id = i % num_gpus
        fingerprint_model(name, hf_id, gpu_id)

    # Global summary
    log.info("\n" + "=" * 70)
    log.info("FINGERPRINTING COMPLETE — GLOBAL TOP LAYERS:")
    all_layers = []
    for name in MODELS:
        top_path = os.path.join(FINGERPRINT_DIR, f"{name}_top_layers.json")
        if os.path.exists(top_path):
            with open(top_path) as f:
                layers = json.load(f)
            for entry in layers[:10]:
                entry["model"] = name
                all_layers.append(entry)

    all_layers.sort(key=lambda x: x["reasoning_score"], reverse=True)
    log.info(f"  {'Model':>15}  {'Layer':>6}  {'Score':>8}")
    log.info(f"  {'─' * 35}")
    for entry in all_layers[:20]:
        log.info(f"  {entry['model']:>15}  {entry['layer_idx']:>6}  "
                 f"{entry['reasoning_score']:>8.4f}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
