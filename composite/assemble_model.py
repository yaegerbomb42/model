#!/usr/bin/env python3
"""
assemble_model.py — Build the final composite model from evolutionary results.

Takes the best layer sequence from evolutionary search, the fine-tuned bridges,
and assembles the complete composite model checkpoint. Also exports a config
file that describes how to load and run the model.
"""

import os
import sys
import json
import time
import logging
import gc

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODELS, HIDDEN_DIMS, NUM_LAYERS,
    BACKBONE_PREFIX_LAYERS, BACKBONE_SUFFIX_LAYERS,
    LAYER_CACHE_DIR, BRIDGE_DIR, EVOL_DIR, OUTPUT_DIR,
    DTYPE, SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

torch.manual_seed(SEED)


def main():
    log.info("=" * 70)
    log.info("COMPOSITE MODEL — PHASE 6: Assembly")
    log.info("=" * 70)
    t0 = time.time()

    # Load best layer sequence
    best_seq_path = os.path.join(EVOL_DIR, "best_layer_sequence.json")
    if not os.path.exists(best_seq_path):
        log.error(f"Best sequence not found: {best_seq_path}")
        return

    with open(best_seq_path) as f:
        best_data = json.load(f)
    best_sequence = best_data["sequence"]
    log.info(f"Best sequence: {len(best_sequence)} layers, fitness={best_data['fitness']:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # 1. Build the composite model config
    # ─────────────────────────────────────────────────────────────────────

    composite_config = {
        "architecture": "composite_frankenstein_v1",
        "backbone_model": MODELS["backbone"],
        "backbone_hidden_dim": HIDDEN_DIMS["backbone"],
        "backbone_prefix_layers": BACKBONE_PREFIX_LAYERS,
        "backbone_suffix_layers": BACKBONE_SUFFIX_LAYERS,
        "frontier_sequence": [],
        "bridge_files": [],
        "total_frontier_layers": len(best_sequence),
        "dtype": DTYPE,
    }

    # ─────────────────────────────────────────────────────────────────────
    # 2. Collect all layer files and bridge files
    # ─────────────────────────────────────────────────────────────────────

    all_layer_files = []
    all_bridge_files = []

    for i, entry in enumerate(best_sequence):
        model_name = entry["model_name"]
        layer_idx = entry["layer_idx"]
        hidden_dim = entry["hidden_dim"]

        # Layer file
        layer_file = f"{model_name}_layer_{layer_idx}.pt"
        layer_path = os.path.join(LAYER_CACHE_DIR, layer_file)
        if not os.path.exists(layer_path):
            log.warning(f"Missing layer file: {layer_path}")
            continue

        # Check for fine-tuned bridges first, fall back to Procrustes
        bridge_pre_file = f"bridge_finetuned_backbone_to_{model_name}_{i}.pt"
        bridge_post_file = f"bridge_finetuned_{model_name}_to_backbone_{i}.pt"

        bridge_pre_path = os.path.join(BRIDGE_DIR, bridge_pre_file)
        bridge_post_path = os.path.join(BRIDGE_DIR, bridge_post_file)

        if not os.path.exists(bridge_pre_path):
            # Fall back to Procrustes bridge
            backbone_dim = HIDDEN_DIMS["backbone"]
            bridge_pre_file = f"bridge_backbone_{backbone_dim}_to_{model_name}_{hidden_dim}.pt"
            bridge_pre_path = os.path.join(BRIDGE_DIR, bridge_pre_file)

        if not os.path.exists(bridge_post_path):
            backbone_dim = HIDDEN_DIMS["backbone"]
            bridge_post_file = f"bridge_{model_name}_{hidden_dim}_to_backbone_{backbone_dim}.pt"
            bridge_post_path = os.path.join(BRIDGE_DIR, bridge_post_file)

        frontier_entry = {
            "slot_index": i,
            "model_name": model_name,
            "layer_idx": layer_idx,
            "hidden_dim": hidden_dim,
            "reasoning_score": entry.get("reasoning_score", 0.0),
            "layer_file": layer_file,
            "bridge_pre_file": os.path.basename(bridge_pre_path) if os.path.exists(bridge_pre_path) else None,
            "bridge_post_file": os.path.basename(bridge_post_path) if os.path.exists(bridge_post_path) else None,
        }
        composite_config["frontier_sequence"].append(frontier_entry)

        all_layer_files.append(layer_path)
        if os.path.exists(bridge_pre_path):
            all_bridge_files.append(bridge_pre_path)
        if os.path.exists(bridge_post_path):
            all_bridge_files.append(bridge_post_path)

    # ─────────────────────────────────────────────────────────────────────
    # 3. Bundle all bridges into a single checkpoint
    # ─────────────────────────────────────────────────────────────────────

    log.info("Bundling bridge matrices ...")
    bridge_bundle = {}
    for bp in set(all_bridge_files):
        if os.path.exists(bp):
            key = os.path.basename(bp).replace(".pt", "")
            bridge_bundle[key] = torch.load(bp, map_location="cpu")

    bridge_bundle_path = os.path.join(OUTPUT_DIR, "bridges_bundle.pt")
    torch.save(bridge_bundle, bridge_bundle_path)
    bundle_size_mb = os.path.getsize(bridge_bundle_path) / (1024 * 1024)
    log.info(f"Bridge bundle: {len(bridge_bundle)} bridges, {bundle_size_mb:.1f} MB")

    # ─────────────────────────────────────────────────────────────────────
    # 4. Save composite config
    # ─────────────────────────────────────────────────────────────────────

    config_path = os.path.join(OUTPUT_DIR, "composite_config.json")
    with open(config_path, "w") as f:
        json.dump(composite_config, f, indent=2)
    log.info(f"Composite config saved to {config_path}")

    # ─────────────────────────────────────────────────────────────────────
    # 5. Create a manifest of all required files for local deployment
    # ─────────────────────────────────────────────────────────────────────

    manifest = {
        "description": "Composite Frankenstein Model v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backbone_model_hf": MODELS["backbone"],
        "files_to_download": {
            "config": "output/composite_config.json",
            "bridges": "output/bridges_bundle.pt",
            "layer_files": [os.path.basename(f) for f in all_layer_files],
        },
        "total_size_gb": {
            "backbone": 14.0,
            "bridges": bundle_size_mb / 1024,
            "layers": sum(
                os.path.getsize(f) for f in all_layer_files if os.path.exists(f)
            ) / (1024**3),
        },
        "runtime_requirements": {
            "min_ram_gb": 16,
            "min_ssd_gb": 200,
            "recommended_vram_gb": 20,
        },
    }
    manifest["total_size_gb"]["total"] = sum(manifest["total_size_gb"].values())

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────

    elapsed = time.time() - t0
    log.info("\n" + "=" * 70)
    log.info("ASSEMBLY COMPLETE")
    log.info(f"  Config:       {config_path}")
    log.info(f"  Bridges:      {bridge_bundle_path} ({bundle_size_mb:.1f} MB)")
    log.info(f"  Manifest:     {manifest_path}")
    log.info(f"  Layer files:  {len(all_layer_files)} files in {LAYER_CACHE_DIR}")
    log.info(f"  Total size:   {manifest['total_size_gb']['total']:.1f} GB")
    log.info(f"  Elapsed:      {elapsed:.0f}s")
    log.info("")
    log.info("TO RUN LOCALLY:")
    log.info(f"  1. Download backbone: huggingface-cli download {MODELS['backbone']}")
    log.info(f"  2. Copy layer_cache/ and output/ to your Mac")
    log.info(f"  3. Run: python composite/run_inference.py --prompt 'your question'")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
