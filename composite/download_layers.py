#!/usr/bin/env python3
"""
download_layers.py — Download frontier models and extract reasoning layers.

For each model in config.MODELS, downloads via HuggingFace Hub, then extracts
ONLY the middle reasoning layers (the ones with highest reasoning signal) and
saves each layer's state_dict as a separate .pt file.

Runs on Lambda cluster — expects multi-GPU with large VRAM.
"""

import os
import sys
import json
import time
import logging
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODELS, HIDDEN_DIMS, NUM_LAYERS, LAYER_RANGES,
    LAYER_CACHE_DIR, QUANTIZE_INT8, DTYPE, NUM_GPUS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Download model weights (parallel HF downloads)
# ─────────────────────────────────────────────────────────────────────────────

def download_model(name: str, hf_id: str) -> str:
    """Download a single model from HuggingFace Hub. Returns cache path."""
    from huggingface_hub import snapshot_download

    log.info(f"[{name}] Starting download of {hf_id} ...")
    t0 = time.time()
    try:
        cache_path = snapshot_download(
            repo_id=hf_id,
            ignore_patterns=["*.gguf", "*.md", "*.txt", "original/*"],
        )
        elapsed = time.time() - t0
        log.info(f"[{name}] Download complete in {elapsed:.0f}s → {cache_path}")
        return cache_path
    except Exception as e:
        log.error(f"[{name}] Download FAILED: {e}")
        return None


def download_all_models() -> dict:
    """Download all frontier models in parallel. Returns {name: cache_path}."""
    results = {}
    max_workers = min(len(MODELS), 4)  # limit parallel downloads to avoid bandwidth saturation

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_model, name, hf_id): name
            for name, hf_id in MODELS.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                path = future.result()
                if path:
                    results[name] = path
            except Exception as e:
                log.error(f"[{name}] Unexpected error: {e}")

    log.info(f"Successfully downloaded {len(results)}/{len(MODELS)} models.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Extract middle reasoning layers as individual .pt files
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_accessor(model):
    """Return the list of transformer layers from the model, handling different architectures."""
    # Most models (Llama, Qwen, Gemma, Mistral) use model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Fallback: some models use model.transformer.h (GPT-2 style)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot find layer accessor for model type: {type(model)}")


def extract_layers_for_model(name: str, hf_id: str, gpu_id: int):
    """Load a model, extract its middle reasoning layers, save each as .pt file."""
    from transformers import AutoModelForCausalLM, AutoConfig

    layer_start, layer_end = LAYER_RANGES[name]
    expected_hidden = HIDDEN_DIMS[name]
    expected_nlayers = NUM_LAYERS[name]

    # Check if layers already extracted
    existing = [
        f for f in os.listdir(LAYER_CACHE_DIR)
        if f.startswith(f"{name}_layer_") and f.endswith(".pt")
    ]
    if len(existing) >= (layer_end - layer_start):
        log.info(f"[{name}] {len(existing)} layers already cached. Skipping extraction.")
        return

    log.info(f"[{name}] Loading model onto GPU {gpu_id} for layer extraction ...")
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    try:
        # Load model — use INT8 quantization if configured
        load_kwargs = {
            "torch_dtype": getattr(torch, DTYPE),
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if QUANTIZE_INT8 and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = {"": gpu_id}
        else:
            load_kwargs["device_map"] = {"": device}

        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        model.eval()

        layers = get_layer_accessor(model)
        actual_nlayers = len(layers)
        log.info(f"[{name}] Loaded. {actual_nlayers} layers found. "
                 f"Extracting layers {layer_start}-{layer_end - 1} ...")

        # Verify architecture
        if actual_nlayers != expected_nlayers:
            log.warning(f"[{name}] Expected {expected_nlayers} layers, got {actual_nlayers}. "
                        f"Adjusting range.")
            layer_end = min(layer_end, actual_nlayers)

        # Extract and save each layer
        for idx in range(layer_start, layer_end):
            out_path = os.path.join(LAYER_CACHE_DIR, f"{name}_layer_{idx}.pt")
            if os.path.exists(out_path):
                log.info(f"  [{name}] Layer {idx} already cached.")
                continue

            layer = layers[idx]
            # Move to CPU before saving (saves VRAM, safetensors on CPU)
            state = {k: v.detach().cpu() for k, v in layer.state_dict().items()}
            torch.save(state, out_path)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            log.info(f"  [{name}] Layer {idx} → {out_path} ({size_mb:.1f} MB)")

        # Save metadata
        meta = {
            "model_name": name,
            "hf_id": hf_id,
            "hidden_dim": expected_hidden,
            "num_layers": actual_nlayers,
            "extracted_range": [layer_start, layer_end],
            "num_extracted": layer_end - layer_start,
            "architecture": model.config.model_type,
            "intermediate_size": getattr(model.config, "intermediate_size", None),
            "num_attention_heads": getattr(model.config, "num_attention_heads", None),
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
        }
        meta_path = os.path.join(LAYER_CACHE_DIR, f"{name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"[{name}] Metadata saved to {meta_path}")

    except Exception as e:
        log.error(f"[{name}] Layer extraction FAILED: {e}", exc_info=True)
    finally:
        # Free GPU memory aggressively
        try:
            del model
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log.info(f"[{name}] Extraction complete in {elapsed:.0f}s")


def extract_all_layers():
    """Extract reasoning layers from all models, distributing across GPUs."""
    model_list = list(MODELS.items())
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    log.info(f"Extracting layers using {num_gpus} GPU(s) ...")

    for i, (name, hf_id) in enumerate(model_list):
        gpu_id = i % num_gpus
        extract_layers_for_model(name, hf_id, gpu_id)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("COMPOSITE MODEL — PHASE 1: Download & Extract Frontier Layers")
    log.info("=" * 70)

    # Step 1: Download all models in parallel
    log.info("STEP 1: Downloading models from HuggingFace ...")
    download_all_models()

    # Step 2: Extract reasoning layers
    log.info("STEP 2: Extracting middle reasoning layers ...")
    extract_all_layers()

    # Summary
    layer_files = [f for f in os.listdir(LAYER_CACHE_DIR) if f.endswith(".pt")]
    meta_files = [f for f in os.listdir(LAYER_CACHE_DIR) if f.endswith(".json")]
    total_size_gb = sum(
        os.path.getsize(os.path.join(LAYER_CACHE_DIR, f))
        for f in layer_files
    ) / (1024**3)

    log.info("=" * 70)
    log.info(f"EXTRACTION COMPLETE:")
    log.info(f"  Layer files: {len(layer_files)}")
    log.info(f"  Metadata files: {len(meta_files)}")
    log.info(f"  Total size: {total_size_gb:.1f} GB")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
