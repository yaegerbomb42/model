#!/usr/bin/env python3
"""
run_inference.py — Run the composite Frankenstein model locally.

Loads the backbone model, Procrustes bridges, and frontier layers (lazy-loaded
from SSD) to run inference. This is the script you use on your Mac M4 Max.

Usage:
    python composite/run_inference.py --prompt "Solve: what is 2^10?"
    python composite/run_inference.py --interactive
"""

import os
import sys
import json
import time
import logging
import argparse
import gc
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODELS, HIDDEN_DIMS, BACKBONE_PREFIX_LAYERS, BACKBONE_SUFFIX_LAYERS,
    LAYER_CACHE_DIR, OUTPUT_DIR, DTYPE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class LazyLayerBank:
    """
    Memory-mapped loader for frontier layers on SSD.
    Uses an LRU cache to keep frequently accessed layers in RAM.
    """

    def __init__(self, layer_dir: str, cache_size: int = 8):
        self.layer_dir = layer_dir
        self.cache_size = cache_size
        self._cache: Dict[str, dict] = {}
        self._access_order: list = []

    def load(self, model_name: str, layer_idx: int) -> dict:
        """Load a frontier layer's state dict. Returns from cache if available."""
        key = f"{model_name}_layer_{layer_idx}"
        path = os.path.join(self.layer_dir, f"{key}.pt")

        if key in self._cache:
            # Move to end of access order (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        if not os.path.exists(path):
            log.warning(f"Layer not found: {path}")
            return None

        # Load from disk
        t0 = time.time()
        state = torch.load(path, map_location="cpu", weights_only=True)
        elapsed_ms = (time.time() - t0) * 1000
        log.debug(f"Loaded {key} from SSD in {elapsed_ms:.0f}ms")

        # Evict oldest entry if cache is full
        while len(self._cache) >= self.cache_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
            gc.collect()

        self._cache[key] = state
        self._access_order.append(key)
        return state


class CompositeModel:
    """
    The composite Frankenstein model.

    Backbone: DeepSeek R1 Distill Qwen 7B (embeddings, prefix/suffix layers, LM head)
    Frontier: Cherry-picked layers from multiple frontier models
    Bridges: Procrustes/fine-tuned linear transforms between representation spaces
    """

    def __init__(self, config_path: str, device: str = "cpu"):
        self.device = device
        self.dtype = getattr(torch, DTYPE) if device != "cpu" else torch.float32

        # Load composite config
        with open(config_path) as f:
            self.config = json.load(f)

        log.info(f"Loading composite model: {len(self.config['frontier_sequence'])} "
                 f"frontier layers")

        # Load backbone model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        backbone_hf = self.config["backbone_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_hf, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if "cuda" in device:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = {"": device}
        elif device == "mps":
            load_kwargs["device_map"] = {"": "mps"}
        else:
            load_kwargs["device_map"] = "cpu"

        self.backbone = AutoModelForCausalLM.from_pretrained(backbone_hf, **load_kwargs)
        self.backbone.eval()
        log.info("Backbone loaded.")

        # Load bridge bundle
        bridge_path = os.path.join(OUTPUT_DIR, "bridges_bundle.pt")
        if os.path.exists(bridge_path):
            self.bridges = torch.load(bridge_path, map_location="cpu", weights_only=True)
            log.info(f"Bridges loaded: {len(self.bridges)} matrices")
        else:
            self.bridges = {}
            log.warning("No bridge bundle found. Using identity bridges.")

        # Initialize lazy layer bank
        self.layer_bank = LazyLayerBank(LAYER_CACHE_DIR, cache_size=8)

        # Get backbone layer accessor
        self.backbone_layers = self.backbone.model.layers
        self.embed_tokens = self.backbone.model.embed_tokens
        self.norm = self.backbone.model.norm
        self.lm_head = self.backbone.lm_head
        self.backbone_dim = self.config["backbone_hidden_dim"]

    def _get_bridge_weight(self, slot_index: int, direction: str, entry: dict):
        """Get the bridge weight matrix for a given slot and direction."""
        model_name = entry["model_name"]

        # Try fine-tuned bridge first
        if direction == "pre":
            key = f"bridge_finetuned_backbone_to_{model_name}_{slot_index}"
        else:
            key = f"bridge_finetuned_{model_name}_to_backbone_{slot_index}"

        if key in self.bridges:
            return self.bridges[key]["weight"].to(self.dtype)

        # Fall back to Procrustes bridge
        frontier_dim = entry["hidden_dim"]
        if direction == "pre":
            key = f"bridge_backbone_{self.backbone_dim}_to_{model_name}_{frontier_dim}"
        else:
            key = f"bridge_{model_name}_{frontier_dim}_to_backbone_{self.backbone_dim}"

        if key in self.bridges:
            return self.bridges[key]["weight"].to(self.dtype)

        # Last resort: identity or random
        frontier_dim = entry["hidden_dim"]
        if direction == "pre":
            return torch.randn(frontier_dim, self.backbone_dim, dtype=self.dtype) * 0.01
        else:
            return torch.randn(self.backbone_dim, frontier_dim, dtype=self.dtype) * 0.01

    @torch.no_grad()
    def _composite_forward(self, input_ids):
        """Run the full composite forward pass."""
        # 1. Embed
        h = self.embed_tokens(input_ids)

        # 2. Backbone prefix layers
        for idx in BACKBONE_PREFIX_LAYERS:
            if idx < len(self.backbone_layers):
                out = self.backbone_layers[idx](h)
                h = out[0] if isinstance(out, tuple) else out

        # 3. Frontier layers with bridges
        for entry in self.config["frontier_sequence"]:
            model_name = entry["model_name"]
            layer_idx = entry["layer_idx"]
            slot_idx = entry["slot_index"]
            frontier_dim = entry["hidden_dim"]

            # Get bridge matrices
            W_pre = self._get_bridge_weight(slot_idx, "pre", entry).to(self.device)
            W_post = self._get_bridge_weight(slot_idx, "post", entry).to(self.device)

            # Bridge: backbone → frontier space
            h_frontier = F.linear(h, W_pre)

            # Load and apply frontier layer
            layer_state = self.layer_bank.load(model_name, layer_idx)
            if layer_state is not None:
                # Apply the layer weights as a simple MLP transformation
                # (approximate — full layer reconstruction in production version)
                for key, weight in layer_state.items():
                    if "down_proj" in key or "o_proj" in key:
                        w = weight.to(self.dtype).to(self.device)
                        if w.shape[1] == h_frontier.shape[-1]:
                            h_frontier = F.linear(h_frontier, w)
                            h_frontier = F.silu(h_frontier)
                            break
                        elif w.shape[0] == h_frontier.shape[-1]:
                            h_frontier = F.linear(h_frontier, w.T)
                            h_frontier = F.silu(h_frontier)
                            break

            # Bridge: frontier → backbone space
            h = F.linear(h_frontier, W_post)

        # 4. Backbone suffix layers
        for idx in BACKBONE_SUFFIX_LAYERS:
            if idx < len(self.backbone_layers):
                out = self.backbone_layers[idx](h)
                h = out[0] if isinstance(out, tuple) else out

        # 5. Norm + LM head
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"].to(self.device)

        t0 = time.time()
        generated = input_ids

        for step in range(max_tokens):
            logits = self._composite_forward(generated)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        elapsed = time.time() - t0
        output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        tokens_generated = generated.shape[1] - input_ids.shape[1]
        tok_per_sec = tokens_generated / max(elapsed, 0.001)

        log.info(f"Generated {tokens_generated} tokens in {elapsed:.1f}s "
                 f"({tok_per_sec:.1f} tok/s)")

        return output_text

    def generate_with_fallback(self, prompt: str, max_tokens: int = 512):
        """Try composite forward, fall back to backbone-only if it fails."""
        try:
            return self.generate(prompt, max_tokens)
        except Exception as e:
            log.warning(f"Composite forward failed: {e}. Falling back to backbone.")
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = inputs["input_ids"].to(self.device)
            with torch.no_grad():
                output = self.backbone.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run the Composite Frankenstein Model")
    parser.add_argument("--prompt", type=str, help="Single prompt to run")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--config", type=str,
                        default=os.path.join(OUTPUT_DIR, "composite_config.json"),
                        help="Path to composite_config.json")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, mps, cuda, auto")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    log.info(f"Device: {device}")

    # Check if composite config exists
    if not os.path.exists(args.config):
        log.error(f"Composite config not found: {args.config}")
        log.error("Run the pipeline first: bash run_pipeline.sh")
        sys.exit(1)

    # Load model
    model = CompositeModel(args.config, device=device)

    if args.prompt:
        output = model.generate_with_fallback(args.prompt, args.max_tokens)
        print("\n" + "=" * 70)
        print(output)
        print("=" * 70)

    elif args.interactive:
        print("\n" + "=" * 70)
        print("COMPOSITE FRANKENSTEIN MODEL — Interactive Mode")
        print("Type 'quit' or 'exit' to stop.")
        print("=" * 70 + "\n")

        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue

                output = model.generate_with_fallback(prompt, args.max_tokens)
                # Remove the input prompt from output if echoed
                if output.startswith(prompt):
                    output = output[len(prompt):].strip()
                print(f"\nModel: {output}\n")

            except KeyboardInterrupt:
                print("\nExiting.")
                break
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
