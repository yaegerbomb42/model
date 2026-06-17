#!/usr/bin/env python3
"""
evolutionary_search.py — Evolutionary algorithm to find optimal layer ordering.

Uses a genetic algorithm to search over possible sequences of frontier layers
from different models. Fitness = accuracy on reasoning benchmarks using the
composite layer sequence with Procrustes bridges.

NO gradient training. Just forward passes + selection + mutation.
"""

import os
import sys
import json
import time
import random
import logging
import gc
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODELS, HIDDEN_DIMS, NUM_LAYERS, LAYER_RANGES,
    BACKBONE_PREFIX_LAYERS, BACKBONE_SUFFIX_LAYERS,
    NUM_FRONTIER_SLOTS, LAYER_CACHE_DIR, BRIDGE_DIR,
    FINGERPRINT_DIR, EVOL_DIR, SEED,
    EVOL_POPULATION_SIZE, EVOL_GENERATIONS,
    EVOL_MUTATION_RATE, EVOL_CROSSOVER_RATE,
    EVOL_ELITE_FRACTION, EVOL_FITNESS_QUESTIONS,
    EVOL_PARALLEL_WORKERS, DTYPE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerSlot:
    model_name: str
    layer_idx: int
    hidden_dim: int
    reasoning_score: float = 0.0

@dataclass
class Chromosome:
    slots: List[LayerSlot]
    fitness: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Build the pool of available layers
# ─────────────────────────────────────────────────────────────────────────────

def build_layer_pool(top_k_per_model: int = 20) -> List[LayerSlot]:
    """Load fingerprint rankings and build a pool of the best layers."""
    pool = []

    for name in MODELS:
        top_path = os.path.join(FINGERPRINT_DIR, f"{name}_top_layers.json")
        if not os.path.exists(top_path):
            log.warning(f"[{name}] No fingerprint found at {top_path}. "
                        f"Using default layer range.")
            # Fallback: use middle layers with uniform score
            start, end = LAYER_RANGES[name]
            for idx in range(start, min(start + top_k_per_model, end)):
                pool.append(LayerSlot(
                    model_name=name,
                    layer_idx=idx,
                    hidden_dim=HIDDEN_DIMS[name],
                    reasoning_score=0.5,
                ))
            continue

        with open(top_path) as f:
            ranked = json.load(f)

        for entry in ranked[:top_k_per_model]:
            pool.append(LayerSlot(
                model_name=name,
                layer_idx=entry["layer_idx"],
                hidden_dim=HIDDEN_DIMS[name],
                reasoning_score=entry["reasoning_score"],
            ))

    log.info(f"Layer pool: {len(pool)} candidate layers from {len(MODELS)} models")
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Population initialization
# ─────────────────────────────────────────────────────────────────────────────

def create_initial_population(pool: List[LayerSlot], pop_size: int) -> List[Chromosome]:
    """Create initial population of chromosomes."""
    population = []

    # Chromosome 1: top-scored layers in order of reasoning score
    sorted_pool = sorted(pool, key=lambda x: x.reasoning_score, reverse=True)
    top_chromosome = Chromosome(slots=copy.deepcopy(sorted_pool[:NUM_FRONTIER_SLOTS]))
    population.append(top_chromosome)

    # Chromosome 2: diverse — pick top layer from each model in round-robin
    diverse_slots = []
    model_pools = {}
    for slot in sorted_pool:
        if slot.model_name not in model_pools:
            model_pools[slot.model_name] = []
        model_pools[slot.model_name].append(slot)

    model_names = list(model_pools.keys())
    model_idx = 0
    model_counters = {m: 0 for m in model_names}
    while len(diverse_slots) < NUM_FRONTIER_SLOTS and model_names:
        name = model_names[model_idx % len(model_names)]
        if model_counters[name] < len(model_pools[name]):
            diverse_slots.append(copy.deepcopy(model_pools[name][model_counters[name]]))
            model_counters[name] += 1
        model_idx += 1
        if all(model_counters[m] >= len(model_pools[m]) for m in model_names):
            break
    while len(diverse_slots) < NUM_FRONTIER_SLOTS:
        diverse_slots.append(copy.deepcopy(random.choice(pool)))
    population.append(Chromosome(slots=diverse_slots[:NUM_FRONTIER_SLOTS]))

    # Remaining chromosomes: random selections with slight bias toward high-scoring
    weights = [s.reasoning_score + 0.1 for s in pool]
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]

    while len(population) < pop_size:
        import numpy as np
        indices = np.random.choice(len(pool), size=NUM_FRONTIER_SLOTS, replace=False, p=probs)
        slots = [copy.deepcopy(pool[i]) for i in indices]
        population.append(Chromosome(slots=slots))

    log.info(f"Initial population: {len(population)} chromosomes")
    return population


# ─────────────────────────────────────────────────────────────────────────────
# Fitness evaluation — the core of the search
# ─────────────────────────────────────────────────────────────────────────────

class CompositeForward(nn.Module):
    """
    A lightweight composite model that chains:
    backbone_prefix → [bridge → frontier_layer → bridge]* → backbone_suffix → lm_head

    All frontier layers and bridges are loaded from disk on construction.
    """

    def __init__(self, chromosome: Chromosome, backbone_model, device: str):
        super().__init__()
        self.device = device
        self.dtype = getattr(torch, DTYPE)

        # Extract backbone components
        self.embed_tokens = backbone_model.model.embed_tokens
        self.backbone_layers = backbone_model.model.layers
        self.norm = backbone_model.model.norm
        self.lm_head = backbone_model.lm_head
        self.backbone_dim = HIDDEN_DIMS["backbone"]

        # Load frontier layers and bridges for this chromosome
        self.frontier_layers = nn.ModuleList()
        self.pre_bridges = nn.ParameterList()   # backbone_dim → frontier_dim
        self.post_bridges = nn.ParameterList()  # frontier_dim → backbone_dim

        for slot in chromosome.slots:
            # Load frontier layer state dict
            layer_path = os.path.join(
                LAYER_CACHE_DIR,
                f"{slot.model_name}_layer_{slot.layer_idx}.pt"
            )
            if not os.path.exists(layer_path):
                log.warning(f"Layer file missing: {layer_path}. Using identity.")
                self.frontier_layers.append(nn.Identity())
                self.pre_bridges.append(nn.Parameter(
                    torch.eye(self.backbone_dim), requires_grad=False))
                self.post_bridges.append(nn.Parameter(
                    torch.eye(self.backbone_dim), requires_grad=False))
                continue

            # We can't easily reconstruct the exact layer class without the model.
            # Instead, store the state dict and use backbone layers as a proxy.
            # For the evolutionary search, we approximate by loading the state dict
            # into a backbone-shaped layer and applying the bridge transforms.
            # This is a simplification — the full assemble_model.py will do it correctly.

            # Load bridge: backbone → frontier model
            frontier_dim = slot.hidden_dim
            bridge_pre_path = self._find_bridge(self.backbone_dim, frontier_dim, slot.model_name)
            bridge_post_path = self._find_bridge(frontier_dim, self.backbone_dim, slot.model_name)

            if bridge_pre_path and os.path.exists(bridge_pre_path):
                bridge_pre = torch.load(bridge_pre_path, map_location="cpu")
                W_pre = bridge_pre["weight"].to(self.dtype)
            else:
                W_pre = torch.eye(min(self.backbone_dim, frontier_dim), dtype=self.dtype)
                if self.backbone_dim != frontier_dim:
                    W_pre = torch.randn(frontier_dim, self.backbone_dim, dtype=self.dtype) * 0.01

            if bridge_post_path and os.path.exists(bridge_post_path):
                bridge_post = torch.load(bridge_post_path, map_location="cpu")
                W_post = bridge_post["weight"].to(self.dtype)
            else:
                W_post = torch.eye(min(frontier_dim, self.backbone_dim), dtype=self.dtype)
                if self.backbone_dim != frontier_dim:
                    W_post = torch.randn(self.backbone_dim, frontier_dim, dtype=self.dtype) * 0.01

            self.pre_bridges.append(nn.Parameter(W_pre, requires_grad=False))
            self.post_bridges.append(nn.Parameter(W_post, requires_grad=False))

            # For the search phase, we just use a simple linear proxy
            # (actual layer loading happens in assemble_model.py)
            layer_sd = torch.load(layer_path, map_location="cpu")
            # Use a simple feedforward proxy for speed during search
            proxy = nn.Sequential(
                nn.Linear(frontier_dim, frontier_dim, bias=False),
                nn.SiLU(),
                nn.Linear(frontier_dim, frontier_dim, bias=False),
            )
            # Initialize proxy with pieces from the actual layer weights if possible
            for key, val in layer_sd.items():
                if "down_proj" in key and val.shape[0] == frontier_dim:
                    proxy[2].weight.data = val[:frontier_dim, :frontier_dim].to(self.dtype)
                    break
            self.frontier_layers.append(proxy)

        self.to(device)

    def _find_bridge(self, dim_from: int, dim_to: int, model_name: str) -> Optional[str]:
        """Find a bridge file for the given dimension transition."""
        # Try exact match first
        for fname in os.listdir(BRIDGE_DIR) if os.path.isdir(BRIDGE_DIR) else []:
            if (f"_{dim_from}_to_" in fname and f"_to_{model_name}_{dim_to}" in fname):
                return os.path.join(BRIDGE_DIR, fname)
            if (f"_{dim_from}_to_" in fname and f"_{dim_to}.pt" in fname):
                return os.path.join(BRIDGE_DIR, fname)
        # Try backbone→model bridge
        for fname in os.listdir(BRIDGE_DIR) if os.path.isdir(BRIDGE_DIR) else []:
            if f"backbone_{dim_from}_to_{model_name}_{dim_to}" in fname:
                return os.path.join(BRIDGE_DIR, fname)
            if f"{model_name}_{dim_from}_to_backbone_{dim_to}" in fname:
                return os.path.join(BRIDGE_DIR, fname)
        return None

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        """Run the composite forward pass."""
        # 1. Embed
        h = self.embed_tokens(input_ids)

        # 2. Backbone prefix layers
        for idx in BACKBONE_PREFIX_LAYERS:
            if idx < len(self.backbone_layers):
                h = self.backbone_layers[idx](h)[0] if isinstance(
                    self.backbone_layers[idx](h), tuple) else self.backbone_layers[idx](h)

        # 3. Frontier layers with bridges
        for i, (layer, W_pre, W_post) in enumerate(
            zip(self.frontier_layers, self.pre_bridges, self.post_bridges)
        ):
            # Bridge: backbone space → frontier space
            h_proj = F.linear(h, W_pre)
            # Apply frontier layer (or proxy)
            h_frontier = layer(h_proj) if not isinstance(layer, nn.Identity) else h_proj
            # Bridge: frontier space → backbone space
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


def extract_answer_number(text: str) -> Optional[float]:
    """Extract the final numerical answer from a generated response."""
    import re
    # Look for #### pattern (GSM8K format)
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', text)
    if match:
        return float(match.group(1).replace(",", ""))
    # Fallback: last number in text
    numbers = re.findall(r'[+-]?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def evaluate_chromosome(
    chromosome: Chromosome,
    backbone_model,
    tokenizer,
    eval_data: list,
    device: str,
) -> float:
    """Evaluate a chromosome's fitness on the reasoning benchmark."""
    try:
        composite = CompositeForward(chromosome, backbone_model, device)
        composite.eval()

        correct = 0
        total = 0

        for item in eval_data:
            question = item["question"]
            true_answer = item["answer"]

            # Extract ground truth number
            true_number = extract_answer_number(true_answer)
            if true_number is None:
                continue

            prompt = f"Question: {question}\nAnswer: Let me solve this step by step.\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(device)

            # Generate a few tokens
            with torch.no_grad(), autocast(dtype=getattr(torch, DTYPE)):
                try:
                    logits = composite(input_ids)
                    # Simple greedy generation for 100 tokens
                    generated = input_ids
                    for _ in range(100):
                        logits = composite(generated)
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=1)
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                        if generated.shape[1] > 512:
                            break

                    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    pred_number = extract_answer_number(output_text)

                    if pred_number is not None and abs(pred_number - true_number) < 0.01:
                        correct += 1
                except Exception as e:
                    log.debug(f"Forward pass error: {e}")

                total += 1

            del input_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fitness = correct / max(total, 1)
        del composite
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return fitness

    except Exception as e:
        log.error(f"Chromosome evaluation failed: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Genetic operators
# ─────────────────────────────────────────────────────────────────────────────

def crossover(parent_a: Chromosome, parent_b: Chromosome) -> Chromosome:
    """Single-point crossover between two chromosomes."""
    point = random.randint(1, NUM_FRONTIER_SLOTS - 1)
    child_slots = (
        copy.deepcopy(parent_a.slots[:point]) +
        copy.deepcopy(parent_b.slots[point:])
    )
    return Chromosome(slots=child_slots)


def mutate(chromosome: Chromosome, pool: List[LayerSlot]) -> Chromosome:
    """Randomly replace one slot with a different layer from the pool."""
    new_chrom = Chromosome(slots=copy.deepcopy(chromosome.slots))
    idx = random.randint(0, len(new_chrom.slots) - 1)
    new_chrom.slots[idx] = copy.deepcopy(random.choice(pool))
    return new_chrom


# ─────────────────────────────────────────────────────────────────────────────
# Main evolutionary loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("COMPOSITE MODEL — PHASE 4: Evolutionary Layer Search")
    log.info("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load backbone model (stays in memory throughout)
    log.info("Loading backbone model ...")
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
    log.info("Backbone loaded.")

    # Load evaluation data
    log.info("Loading evaluation data ...")
    ds = load_dataset("gsm8k", "main", split="test")
    eval_data = [{"question": row["question"], "answer": row["answer"]}
                 for row in ds.select(range(min(EVOL_FITNESS_QUESTIONS, len(ds))))]
    log.info(f"Eval data: {len(eval_data)} questions")

    # Build layer pool
    pool = build_layer_pool(top_k_per_model=20)

    # Initialize population
    population = create_initial_population(pool, EVOL_POPULATION_SIZE)

    # Evolution history
    history = []
    best_ever = None

    for gen in range(EVOL_GENERATIONS):
        t0 = time.time()
        log.info(f"\n{'─' * 50}")
        log.info(f"GENERATION {gen + 1}/{EVOL_GENERATIONS}")

        # Evaluate fitness for chromosomes that haven't been evaluated
        for i, chrom in enumerate(population):
            if chrom.fitness == 0.0:
                log.info(f"  Evaluating chromosome {i + 1}/{len(population)} ...")
                chrom.fitness = evaluate_chromosome(
                    chrom, backbone_model, tokenizer, eval_data, device
                )
                log.info(f"  Chromosome {i + 1} fitness: {chrom.fitness:.4f}")

        # Sort by fitness
        population.sort(key=lambda c: c.fitness, reverse=True)

        best = population[0]
        mean_fit = sum(c.fitness for c in population) / len(population)

        if best_ever is None or best.fitness > best_ever.fitness:
            best_ever = copy.deepcopy(best)

        gen_log = {
            "generation": gen + 1,
            "best_fitness": best.fitness,
            "mean_fitness": mean_fit,
            "best_sequence": [
                {"model": s.model_name, "layer": s.layer_idx, "score": s.reasoning_score}
                for s in best.slots
            ],
            "elapsed_seconds": time.time() - t0,
        }
        history.append(gen_log)

        log.info(f"  BEST fitness: {best.fitness:.4f}  |  MEAN: {mean_fit:.4f}  "
                 f"|  Time: {gen_log['elapsed_seconds']:.0f}s")
        log.info(f"  Best sequence: {[(s.model_name, s.layer_idx) for s in best.slots[:5]]} ...")

        # Elitism: keep top fraction
        elite_count = max(1, int(EVOL_ELITE_FRACTION * len(population)))
        next_gen = copy.deepcopy(population[:elite_count])

        # Fill rest with crossover + mutation
        while len(next_gen) < EVOL_POPULATION_SIZE:
            r = random.random()

            if r < EVOL_CROSSOVER_RATE:
                # Crossover
                p1 = random.choice(population[:len(population) // 2])
                p2 = random.choice(population[:len(population) // 2])
                child = crossover(p1, p2)
            else:
                # Clone a random parent
                child = copy.deepcopy(random.choice(population[:len(population) // 2]))

            # Mutation
            if random.random() < EVOL_MUTATION_RATE:
                child = mutate(child, pool)

            child.fitness = 0.0  # needs re-evaluation
            next_gen.append(child)

        population = next_gen

    # ─────────────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────────────

    # Save best chromosome
    best_sequence = [
        {
            "model_name": s.model_name,
            "layer_idx": s.layer_idx,
            "hidden_dim": s.hidden_dim,
            "reasoning_score": s.reasoning_score,
        }
        for s in best_ever.slots
    ]

    best_path = os.path.join(EVOL_DIR, "best_layer_sequence.json")
    with open(best_path, "w") as f:
        json.dump({
            "fitness": best_ever.fitness,
            "sequence": best_sequence,
        }, f, indent=2)
    log.info(f"\nBest layer sequence saved to {best_path}")
    log.info(f"Best fitness: {best_ever.fitness:.4f}")

    # Save history
    hist_path = os.path.join(EVOL_DIR, "evolution_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Evolution history saved to {hist_path}")

    # Print final results
    log.info("\n" + "=" * 70)
    log.info("EVOLUTIONARY SEARCH COMPLETE")
    log.info(f"Best fitness: {best_ever.fitness:.4f}")
    log.info("Optimal layer sequence:")
    for i, slot in enumerate(best_ever.slots):
        log.info(f"  Slot {i:>2}: {slot.model_name:>15} layer {slot.layer_idx:>3} "
                 f"(dim={slot.hidden_dim}, reasoning={slot.reasoning_score:.4f})")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
