#!/usr/bin/env python3
"""
benchmark.py — Evaluate the composite model on standard reasoning benchmarks.

Compares: backbone alone vs. composite model on GSM8K, MATH, and optionally HumanEval.
"""

import os
import sys
import json
import re
import time
import logging
import gc

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODELS, HIDDEN_DIMS, BACKBONE_PREFIX_LAYERS, BACKBONE_SUFFIX_LAYERS,
    LAYER_CACHE_DIR, BRIDGE_DIR, EVOL_DIR, OUTPUT_DIR,
    BENCHMARK_DATASETS, MAX_BENCHMARK_ROWS, DTYPE, SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

torch.manual_seed(SEED)


def extract_answer_number(text: str):
    """Extract the final numerical answer from text."""
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    numbers = re.findall(r'[+-]?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def evaluate_model_on_gsm8k(model, tokenizer, device, max_rows=200, model_name="model"):
    """Evaluate a model on GSM8K test set."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(min(max_rows, len(ds))))

    correct = 0
    total = 0
    errors = 0

    for i, row in enumerate(ds):
        question = row["question"]
        true_answer = row["answer"]
        true_number = extract_answer_number(true_answer)
        if true_number is None:
            continue

        prompt = f"Question: {question}\nAnswer: Let me solve this step by step.\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)

        try:
            with torch.no_grad(), autocast(dtype=getattr(torch, DTYPE)):
                generated = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            pred_number = extract_answer_number(output_text)

            if pred_number is not None and abs(pred_number - true_number) < 0.01:
                correct += 1
        except Exception as e:
            errors += 1

        total += 1

        if (i + 1) % 50 == 0:
            acc = correct / max(total, 1)
            log.info(f"  [{model_name}] {i + 1}/{len(ds)}  "
                     f"Accuracy: {acc:.2%} ({correct}/{total})")

        del input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
    }


def main():
    log.info("=" * 70)
    log.info("COMPOSITE MODEL — PHASE 7: Benchmarking")
    log.info("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {}

    # ─────────────────────────────────────────────────────────────────
    # 1. Benchmark the BACKBONE model alone (baseline)
    # ─────────────────────────────────────────────────────────────────

    log.info("\n--- BASELINE: Backbone model alone ---")
    backbone_hf = MODELS["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(backbone_hf, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": getattr(torch, DTYPE),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = {"": 0}

    backbone_model = AutoModelForCausalLM.from_pretrained(backbone_hf, **load_kwargs)
    backbone_model.eval()

    baseline_gsm8k = evaluate_model_on_gsm8k(
        backbone_model, tokenizer, device,
        max_rows=MAX_BENCHMARK_ROWS,
        model_name="backbone"
    )
    results["backbone_gsm8k"] = baseline_gsm8k
    log.info(f"BASELINE GSM8K: {baseline_gsm8k['accuracy']:.2%} "
             f"({baseline_gsm8k['correct']}/{baseline_gsm8k['total']})")

    del backbone_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────
    # 2. Benchmark the COMPOSITE model
    # ─────────────────────────────────────────────────────────────────

    log.info("\n--- COMPOSITE MODEL ---")

    # Load composite config
    config_path = os.path.join(OUTPUT_DIR, "composite_config.json")
    if not os.path.exists(config_path):
        log.warning("Composite config not found. Skipping composite benchmark.")
        log.info("Run assemble_model.py first.")
    else:
        with open(config_path) as f:
            composite_config = json.load(f)
        log.info(f"Composite config loaded: {len(composite_config['frontier_sequence'])} "
                 f"frontier layers")

        # For now, benchmark with the backbone + Procrustes bridges
        # (full composite inference requires run_inference.py)
        # This is a placeholder — full composite eval happens in run_inference.py
        results["composite_config"] = {
            "num_frontier_layers": len(composite_config["frontier_sequence"]),
            "models_used": list(set(
                e["model_name"] for e in composite_config["frontier_sequence"]
            )),
        }
        log.info("Composite benchmark requires run_inference.py for full evaluation.")

    # ─────────────────────────────────────────────────────────────────
    # 3. Save results
    # ─────────────────────────────────────────────────────────────────

    results_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("BENCHMARK RESULTS:")
    log.info(f"  Backbone GSM8K accuracy: {baseline_gsm8k['accuracy']:.2%}")
    log.info(f"  Results saved to: {results_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
