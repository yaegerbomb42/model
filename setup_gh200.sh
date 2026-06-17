#!/bin/bash
# setup_gh200.sh - Environment setup and expert initialization for GH200 (Grace Hopper 96GB H100)

set -e

echo "================================================================="
echo "  Setting up GH200 (Grace Hopper) Environment for R2 Agent Model"
echo "================================================================="

# 1. Update OS and Install System Tools
sudo apt-get update && sudo apt-get install -y tmux git curl python3-pip python3-venv

# 2. Set up Python virtual environment
if [ ! -d "venv" ]; then
    echo "[Setup] Creating Python virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 3. Install GPU-accelerated PyTorch and dependencies
echo "[Setup] Installing PyTorch, Transformers, and optimization libraries..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes numpy huggingface_hub safetensors fastapi uvicorn datasets

# 4. Configure Hugging Face token
if [ -n "$HF_TOKEN" ]; then
    echo "[Setup] Configuring HuggingFace credentials..."
    mkdir -p ~/.cache/huggingface
    echo "$HF_TOKEN" > ~/.cache/huggingface/token
fi

# 5. Initialize tmux session for the 70GB Weight SVD extraction
echo "[Setup] Pre-allocating and initializing 67.6 GB virtual experts..."
tmux new-session -d -s dtsg_init 'source venv/bin/activate && python3 -u initialize_frank_experts.py'

echo "================================================================="
echo "  Setup Completed Successfully! ✅"
echo "  The initialization is running in the background."
echo "  Type 'tmux attach -t dtsg_init' to monitor progress."
echo "================================================================="
