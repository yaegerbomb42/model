"""
config.py — Shared configuration for the Composite Model pipeline.

All pipeline scripts import from here. Change values here to change
behaviour across the entire pipeline without touching individual scripts.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
LAYER_CACHE_DIR = os.path.join(BASE_DIR, "layer_cache")   # extracted layer .pt files
BRIDGE_DIR      = os.path.join(BASE_DIR, "bridges")        # Procrustes bridge matrices
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")         # final checkpoint + results
FINGERPRINT_DIR = os.path.join(BASE_DIR, "fingerprints")   # per-layer reasoning scores
EVOL_DIR        = os.path.join(BASE_DIR, "evolution")      # evolutionary search logs

for _d in [LAYER_CACHE_DIR, BRIDGE_DIR, OUTPUT_DIR, FINGERPRINT_DIR, EVOL_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace model IDs
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    # Backbone: tokenizer, embeddings, LM head — the "glue" of the composite model.
    # Already contains distilled R1 reasoning; also our fallback for any layer.
    "backbone":      "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",

    # Frontier reasoning layers — frozen forever, never fine-tuned.
    "deepseek_32b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek_70b":  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "qwen2_5_72b":     "Qwen/Qwen2.5-72B",
    "gemma_27b":     "google/gemma-3-27b-pt",  # pretrained weights preferred
    # "llama_405b":  "meta-llama/Llama-3.1-405B",  # enable if time/VRAM allows
}

# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants per model
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_DIMS = {
    "backbone":      3584,   # Qwen2 7B hidden dim
    "deepseek_32b":  5120,   # Qwen2 32B hidden dim
    "deepseek_70b":  8192,   # Llama 3 70B hidden dim
    "qwen2_5_72b":   8192,   # Qwen2.5 72B hidden dim
    "gemma_27b":     4608,   # Gemma 3 27B hidden dim
}

NUM_LAYERS = {
    "backbone":      28,
    "deepseek_32b":  64,
    "deepseek_70b":  80,
    "qwen2_5_72b":   80,
    "gemma_27b":     62,
}

# Middle 40% of each model = peak reasoning layers (skip syntax + output layers)
LAYER_RANGES = {
    "backbone":      (11, 22),   # layers 11-21 inclusive
    "deepseek_32b":  (26, 51),   # layers 26-50
    "deepseek_70b":  (32, 64),   # layers 32-63
    "qwen2_5_72b":   (32, 64),   # layers 32-63
    "gemma_27b":     (25, 49),   # layers 25-48
}

# ─────────────────────────────────────────────────────────────────────────────
# Composite model design
# ─────────────────────────────────────────────────────────────────────────────
# Backbone layers that are NEVER replaced (keep syntax + output formatting)
BACKBONE_PREFIX_LAYERS  = list(range(0, 8))    # layers 0-7  — early context
BACKBONE_SUFFIX_LAYERS  = list(range(22, 28))  # layers 22-27 — output formatting

# Number of frontier layers to insert between backbone prefix and suffix
NUM_FRONTIER_SLOTS = 16  # the composite "reasoning core" depth

# Router / halting
ROUTING_SPARSITY_K = 2   # frontier layers activated per token (can raise to 4)

# ─────────────────────────────────────────────────────────────────────────────
# Evolutionary search hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
EVOL_POPULATION_SIZE   = 50
EVOL_GENERATIONS       = 20
EVOL_MUTATION_RATE     = 0.15
EVOL_CROSSOVER_RATE    = 0.40
EVOL_ELITE_FRACTION    = 0.10  # top 10% survive each generation unchanged
EVOL_FITNESS_QUESTIONS = 100   # questions per fitness evaluation
EVOL_PARALLEL_WORKERS  = 2     # one per GPU

# ─────────────────────────────────────────────────────────────────────────────
# Bridge (Procrustes alignment) settings
# ─────────────────────────────────────────────────────────────────────────────
PROCRUSTES_CALIBRATION_SAMPLES = 512   # sentences used to compute cross-covariance
BRIDGE_FINETUNE_STEPS          = 300   # gradient steps on bridges after Procrustes init
BRIDGE_FINETUNE_LR             = 5e-4
BRIDGE_FINETUNE_BATCH_SIZE     = 4     # 16→4 to fit V100 32GB with 8-bit backbone

# ─────────────────────────────────────────────────────────────────────────────
# Calibration / benchmark datasets (HuggingFace dataset IDs)
# ─────────────────────────────────────────────────────────────────────────────
CALIBRATION_DATASET   = "openai/gsm8k"          # used for fingerprinting + Procrustes
BENCHMARK_DATASETS    = ["openai/gsm8k", "hendrycks/competition_math", "openai_humaneval"]
MAX_CALIBRATION_ROWS  = 500
MAX_BENCHMARK_ROWS    = 200              # per dataset during evolutionary eval

# ─────────────────────────────────────────────────────────────────────────────
# Training / device
# ─────────────────────────────────────────────────────────────────────────────
DTYPE            = "bfloat16"   # bfloat16 for H100 performance
QUANTIZE_INT8    = True         # load frontier models in INT8 to halve VRAM
NUM_GPUS         = 8            # Lambda 8x H100 SXM4
SEED             = 42
