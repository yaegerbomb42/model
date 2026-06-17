#!/bin/bash
#
# run_pipeline.sh — Master pipeline for the Composite Frankenstein Model.
#
# Runs all 7 phases in sequence on a Lambda 8x A100 cluster.
# Expected runtime: ~2 hours.
#
# Usage:
#   chmod +x run_pipeline.sh
#   bash run_pipeline.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Tee all output to both terminal and log file
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=================================================================="
echo "  COMPOSITE FRANKENSTEIN MODEL — Master Pipeline"
echo "  Started: $(date)"
echo "  Log: ${MASTER_LOG}"
echo "=================================================================="
echo ""

# ──────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────────────────────────────

echo "[PRE-FLIGHT] Checking environment ..."

# Python
python3 --version || { echo "ERROR: python3 not found"; exit 1; }

# GPU
if python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')" 2>/dev/null; then
    echo "[PRE-FLIGHT] CUDA available ✓"
else
    echo "[PRE-FLIGHT] WARNING: No CUDA detected. Running on CPU (very slow)."
fi

# Install dependencies
echo "[PRE-FLIGHT] Installing dependencies ..."
pip install -q torch transformers datasets huggingface_hub bitsandbytes accelerate sentencepiece protobuf scipy numpy 2>&1 | tail -5

echo "[PRE-FLIGHT] All checks passed."
echo ""

# ──────────────────────────────────────────────────────────────────────
# Phase 1: Download & Extract Frontier Layers
# ──────────────────────────────────────────────────────────────────────

PHASE_START=$(date +%s)
echo "=================================================================="
echo "  PHASE 1/7: Download & Extract Frontier Layers"
echo "  Started: $(date)"
echo "=================================================================="

python3 "${SCRIPT_DIR}/download_layers.py" 2>&1 | tee "${LOG_DIR}/phase1_download_${TIMESTAMP}.log"

PHASE_END=$(date +%s)
echo "  Phase 1 completed in $((PHASE_END - PHASE_START))s"
echo ""

# ──────────────────────────────────────────────────────────────────────
# Phase 2: Activation Fingerprinting
# ──────────────────────────────────────────────────────────────────────

PHASE_START=$(date +%s)
echo "=================================================================="
echo "  PHASE 2/7: Activation Fingerprinting"
echo "  Started: $(date)"
echo "=================================================================="

python3 "${SCRIPT_DIR}/activation_fingerprint.py" 2>&1 | tee "${LOG_DIR}/phase2_fingerprint_${TIMESTAMP}.log"

PHASE_END=$(date +%s)
echo "  Phase 2 completed in $((PHASE_END - PHASE_START))s"
echo ""

# ──────────────────────────────────────────────────────────────────────
# Phase 3: Procrustes Alignment
# ──────────────────────────────────────────────────────────────────────

PHASE_START=$(date +%s)
echo "=================================================================="
echo "  PHASE 3/7: Procrustes Alignment"
echo "  Started: $(date)"
echo "=================================================================="

python3 "${SCRIPT_DIR}/procrustes_align.py" 2>&1 | tee "${LOG_DIR}/phase3_procrustes_${TIMESTAMP}.log"

PHASE_END=$(date +%s)
echo "  Phase 3 completed in $((PHASE_END - PHASE_START))s"
echo ""

# ──────────────────────────────────────────────────────────────────────
# Phase 4: Evolutionary Layer Search
# ──────────────────────────────────────────────────────────────────────

PHASE_START=$(date +%s)
echo "=================================================================="
echo "  PHASE 4/7: Evolutionary Layer Search"
echo "  Started: $(date)"
echo "=================================================================="

python3 "${SCRIPT_DIR}/evolutionary_search.py" 2>&1 | tee "${LOG_DIR}/phase4_evolution_${TIMESTAMP}.log"

PHASE_END=$(date +%s)
echo "  Phase 4 completed in $((PHASE_END - PHASE_START))s"
echo ""

# ──────────────────────────────────────────────────────────────────────
# Phase 5: Bridge Matrix Fine-Tuning
# ──────────────────────────────────────────────────────────────────────

PHASE_START=$(date +%s)
echo "=================================================================="
echo "  PHASE 5/7: Bridge Matrix Fine-Tuning"
echo "  Started: $(date)"
echo "=================================================================="

python3 "${SCRIPT_DIR}/bridge_finetune.py" 2>&1 | tee "${LOG_DIR}/phase5_bridge_${TIMESTAMP}.log"

PHASE_END=$(date +%s)
echo "  Phase 5 completed in $((PHASE_END - PHASE_START))s"
echo ""

# ──────────────────────────────────────────────────────────────────────
# Phase 6: Assembly
# ──────────────────────────────────────────────────────────────────────

PHASE_START=$(date +%s)
echo "=================================================================="
echo "  PHASE 6/7: Model Assembly"
echo "  Started: $(date)"
echo "=================================================================="

python3 "${SCRIPT_DIR}/assemble_model.py" 2>&1 | tee "${LOG_DIR}/phase6_assemble_${TIMESTAMP}.log"

PHASE_END=$(date +%s)
echo "  Phase 6 completed in $((PHASE_END - PHASE_START))s"
echo ""

# ──────────────────────────────────────────────────────────────────────
# Phase 7: Benchmarking
# ──────────────────────────────────────────────────────────────────────

PHASE_START=$(date +%s)
echo "=================================================================="
echo "  PHASE 7/7: Benchmarking"
echo "  Started: $(date)"
echo "=================================================================="

python3 "${SCRIPT_DIR}/benchmark.py" 2>&1 | tee "${LOG_DIR}/phase7_benchmark_${TIMESTAMP}.log"

PHASE_END=$(date +%s)
echo "  Phase 7 completed in $((PHASE_END - PHASE_START))s"
echo ""

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

echo "=================================================================="
echo "  PIPELINE COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  Output files:"
echo "    Config:     ${SCRIPT_DIR}/output/composite_config.json"
echo "    Bridges:    ${SCRIPT_DIR}/output/bridges_bundle.pt"
echo "    Manifest:   ${SCRIPT_DIR}/output/manifest.json"
echo "    Benchmarks: ${SCRIPT_DIR}/output/benchmark_results.json"
echo "    Layer seq:  ${SCRIPT_DIR}/evolution/best_layer_sequence.json"
echo ""
echo "  TO COPY RESULTS TO YOUR MAC:"
echo "    scp -r user@lambda-ip:${SCRIPT_DIR}/output/ ~/Desktop/model/composite/output/"
echo "    scp -r user@lambda-ip:${SCRIPT_DIR}/bridges/ ~/Desktop/model/composite/bridges/"
echo "    scp -r user@lambda-ip:${SCRIPT_DIR}/evolution/ ~/Desktop/model/composite/evolution/"
echo "    scp -r user@lambda-ip:${SCRIPT_DIR}/layer_cache/ ~/Desktop/model/composite/layer_cache/"
echo ""
echo "  TO RUN LOCALLY ON YOUR MAC:"
echo "    python composite/run_inference.py --interactive"
echo "=================================================================="
