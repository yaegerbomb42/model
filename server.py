import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import requests
import random
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from transformers import AutoTokenizer

from agent_model import DTSGModel
from dataset_stream import DatasetStreamer

# ── Startup mode selection ────────────────────────────────────────────────────
# Usage:
#   python server.py              → prompts interactively
#   python server.py --mode ce    → CE-only, no teacher (fast start, low RAM)
#   python server.py --mode distill → full KD with gemma4:12b (high RAM)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--mode", choices=["distill", "ce"], default=None)
_parser.add_argument("--model", choices=["35b", "0.5b"], default=None)
_args, _ = _parser.parse_known_args()

if _args.mode is None:
    print("\n" + "="*60)
    print("  DTSG Server — Choose Active Mode")
    print("="*60)
    print("  [1] inference / ce — Normal model use (runs fast, minimal RAM)")
    print("  [2] distill        — Active learning with local teacher (uses high RAM)")
    print("="*60)
    _choice = input("  Select mode [1/2] (default: 1): ").strip()
    TRAINING_MODE = "distill" if _choice == "2" else "ce"
else:
    TRAINING_MODE = _args.mode

if _args.model is None:
    print("\n" + "="*60)
    print("  DTSG Server — Choose Student Base Seed Model")
    print("="*60)
    print("  [1] 0.5b — Qwen2.5-0.5B (lightweight setup, ~940MB download)")
    print("  [2] 35b  — Qwen3.6-35B-A3B (first 4 layers only, ~3GB download, skips rest)")
    print("="*60)
    _model_choice = input("  Select model [1/2] (default: 1): ").strip()
    SELECTED_MODEL = "35b" if _model_choice == "2" else "0.5b"
else:
    SELECTED_MODEL = _args.model

print(f"\n[Server] Starting in '{TRAINING_MODE}' mode with '{SELECTED_MODEL}' base.\n")

app = FastAPI(title="Ultimate PhD Spec DTSG Server")

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
# MPS has a known bug in PyTorch where empty/placeholder parameters cannot be moved to device.
# Fallback to CPU for MPS devices during seed model loading to ensure stability.
if device.type == "mps":
    print("[Server] MPS device detected. Loading seed model parameters on CPU first to avoid placeholder errors.")
print(f"[Server] Device: {device}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
if SELECTED_MODEL == "35b":
    TEACHER_NAME = "Qwen/Qwen3.6-35B-A3B"
else:
    TEACHER_NAME = "Qwen/Qwen2.5-0.5B"

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_SECRET")
if hf_token:
    print("[Server] Authenticated HuggingFace Hub with HF_TOKEN ✅")
else:
    print("[Server] Warning: Running unauthenticated. Rate limits might be lower ⚠️")

tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME, token=hf_token)

# ── Teacher model (only loaded in distill mode) ───────────────────────────────
teacher_model = None
if TRAINING_MODE == "distill":
    print("[Server] Loading gemma4:12b Q4_K_M teacher (Metal)...")
    try:
        from teacher_llama import LlamaCppTeacher
        teacher_model = LlamaCppTeacher(n_gpu_layers=-1, n_ctx=512)
        print("[Server] Teacher ready ✅")
    except Exception as e:
        print(f"[Server] Teacher load failed: {e}")
        print("[Server] Falling back to CE-only mode.")
        TRAINING_MODE = "ce"

# ── Student model ─────────────────────────────────────────────────────────────
student_model = DTSGModel(teacher_model_name=TEACHER_NAME, num_enzymes=6, max_loops=8, teacher_model=None).to(device)
if device.type == "mps":
    # On macOS MPS, we convert the model to bfloat16 to match the seed embeddings datatype
    # and prevent matrix multiplication datatype mismatch assertions inside Metal kernels.
    student_model = student_model.to(torch.bfloat16)
else:
    student_model = student_model.float()
vocab_size = student_model.vocab_size

CHECKPOINT_PATH = "/Users/yaeger/Desktop/model/dtsg_checkpoint.pt"
if os.path.exists(CHECKPOINT_PATH):
    try:
        print("Loading saved student model weights...")
        student_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device), strict=False)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

# Setup Adaptive Parameter Groups
hypernetwork_params = []
gating_params = []
projection_params = []

for name, param in student_model.named_parameters():
    if not param.requires_grad:
        continue
    if "generator" in name or "hyper_net" in name:
        hypernetwork_params.append(param)
    elif "router" in name or "entropy" in name or "value_head" in name or "halting_head" in name:
        gating_params.append(param)
    else:
        projection_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": hypernetwork_params, "lr": 5e-4, "weight_decay": 0.01},
    {"params": gating_params,       "lr": 2e-4, "weight_decay": 0.01},
    {"params": projection_params,   "lr": 1e-4, "weight_decay": 0.01}
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

# Lock for concurrency
model_lock = threading.Lock()

# Global metrics
metrics = {
    "total_tokens_trained": 0,
    "current_loss": 0.0,
    "last_fetched_url": "None",
    "training_history": [],
    "recent_sources": [],
    "is_training": True,
    "policy_reward_ema": -5.0,
    "avg_ponder_steps": 0.0,
    "active_training_text": "None",
    "neuromodulator_level": 1.0
}

# Live multi-source streamer (Wikipedia, Reddit, HN, NewsAPI, Arxiv, FineWeb-Edu)
_streamer = DatasetStreamer(prefetch=12, min_len=80)

# InfoNCE Contrastive Loss helper
def info_nce_loss(features: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    similarity_matrix = torch.matmul(features, features.T)
    exp_sim = torch.exp(similarity_matrix / temperature)
    mask = ~torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    exp_sim_masked = exp_sim * mask.float()
    sum_negatives = torch.sum(exp_sim_masked, dim=-1)
    loss = -torch.log(torch.clamp(torch.diag(exp_sim) / (torch.diag(exp_sim) + sum_negatives), min=1e-10))
    return loss.mean()

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND TRAINING LOOP
# Mode 'distill': full KD (CE + KL + InfoNCE) using gemma4:12b teacher
# Mode 'ce':      CE-only next-token prediction, no teacher needed
# ─────────────────────────────────────────────────────────────────────────────
def background_distillation_loop():
    global metrics
    print(f"[Worker] Background training started (mode={TRAINING_MODE}).")

    alpha = 0.4
    beta = 0.4
    gamma = 0.2
    temperature = 2.0
    seq_len = 16
    batch_size = 4
    reward_momentum = 0.95

    technical_keywords = ["code", "function", "graph", "loss", "optimizer", "math",
                          "logic", "programming", "compile", "structure", "algorithm",
                          "neural", "model", "training", "gradient", "arxiv"]
    creative_keywords   = ["story", "once", "write", "creative", "fiction", "novel",
                           "poetry", "translation", "narrative", "philosophy"]

    while metrics["is_training"]:
        # ── Pull next sample from live multi-source streamer ──────────────────
        try:
            text, source_label = next(_streamer)
        except Exception as e:
            print(f"[Stream] Fetch error: {e}")
            time.sleep(2)
            continue

        metrics["last_fetched_url"]    = source_label
        metrics["active_training_text"] = text[:300]
            
        tokens = tokenizer.encode(text)
        if len(tokens) <= seq_len + 1:
            time.sleep(2)
            continue
            
        x_batches = []
        y_batches = []
        for idx in range(0, len(tokens) - seq_len - 1, seq_len):
            x_batches.append(tokens[idx : idx + seq_len])
            y_batches.append(tokens[idx + 1 : idx + seq_len + 1])
            if len(x_batches) >= batch_size:
                break
                
        if not x_batches:
            time.sleep(2)
            continue
            
        x_tensor = torch.tensor(x_batches, device=device)
        y_tensor = torch.tensor(y_batches, device=device)
        
        try:
            with model_lock:
                optimizer.zero_grad()
                
                # Autocast context setup (BFloat16 for CUDA/CPU, bypassed for MPS to ensure local stability)
                use_amp = device.type in ["cuda", "cpu"]
                autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp else torch.cuda.amp.autocast(enabled=False)
                
                with autocast_ctx:
                    # Student Forward Pass
                    student_logits, s_final, policy_log_probs, s_contrastive, ponder_loss, halt_probs, ach_val, _ = student_model(x_tensor)

                    if TRAINING_MODE == "distill" and teacher_model is not None:
                        # Full KD: CE + KL divergence + InfoNCE contrastive
                        with torch.no_grad():
                            teacher_outputs = teacher_model(x_tensor)
                            teacher_logits  = teacher_outputs.logits.to(device)

                        ce_loss = F.cross_entropy(student_logits.view(-1, vocab_size), y_tensor.view(-1))
                        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                        teacher_probs     = F.softmax(teacher_logits / temperature, dim=-1)
                        kl_loss = F.kl_div(
                            student_log_probs.view(-1, vocab_size),
                            teacher_probs.view(-1, vocab_size),
                            reduction="batchmean"
                        ) * (temperature ** 2)
                        contr_loss   = info_nce_loss(s_contrastive, temperature=0.1)
                        distill_loss = alpha * ce_loss + beta * kl_loss + gamma * contr_loss
                    else:
                        # CE-only: no teacher needed
                        ce_loss      = F.cross_entropy(student_logits.view(-1, vocab_size), y_tensor.view(-1))
                        distill_loss = ce_loss
                        kl_loss      = torch.tensor(0.0)

                    ortho_loss = student_model.get_orthonormal_penalty()
                
                # DPO Gating Policy Rewards
                with torch.no_grad():
                    student_probs = F.softmax(student_logits, dim=-1)
                    vocab_entropy = -torch.sum(student_probs * torch.log(torch.clamp(student_probs, min=1e-10)), dim=-1).mean()
                    
                    step_reward = -vocab_entropy.item() - ce_loss.item()
                    
                    lowered_text = text.lower()
                    # Resolve embedding via list wrapped _embedding[0] and cast values appropriately
                    x_mean = student_model._embedding[0](x_tensor).mean(dim=1).to(s_final.dtype)
                    s_mean = s_final.mean(dim=1)
                    combined_features = torch.cat([x_mean, s_mean], dim=-1)
                    z_target = student_model.enzymes[0].hyper_net(combined_features)
                    diff = z_target.unsqueeze(1) - student_model.enzymes[0].anchor_coords.unsqueeze(0)
                    distances = torch.norm(diff, p=2, dim=-1)
                    weights = F.softmax(-distances, dim=-1)
                    
                    if any(kw in lowered_text for kw in technical_keywords):
                        gating_reward = 3.0 * (weights[:, 6].mean().item() + weights[:, 1].mean().item())
                        step_reward += gating_reward
                        
                    if any(kw in lowered_text for kw in creative_keywords):
                        gating_reward = 3.0 * (weights[:, 3].mean().item() + weights[:, 7].mean().item())
                        step_reward += gating_reward
                    
                    metrics["policy_reward_ema"] = (
                        reward_momentum * metrics["policy_reward_ema"] + (1 - reward_momentum) * step_reward
                    )
                    
                    # Calculate expected halting steps
                    stacked_probs = torch.stack(halt_probs, dim=0) # (num_loops, batch_size)
                    num_loops_run = stacked_probs.shape[0]
                    steps_range = torch.arange(1, num_loops_run + 1, dtype=torch.float, device=device).unsqueeze(1)
                    expected_steps = torch.sum(stacked_probs * steps_range, dim=0).mean().item()
                    metrics["avg_ponder_steps"] = round(expected_steps, 2)
                    metrics["neuromodulator_level"] = round(ach_val, 3)
                    
                advantage = step_reward - metrics["policy_reward_ema"]
                
                # Policy Gradient Loss
                policy_loss = 0.0
                if policy_log_probs:
                    stacked_log_probs = torch.stack(policy_log_probs, dim=0)
                    policy_loss = -torch.mean(stacked_log_probs) * advantage
                    
                # Combined Loss with Orthonormal Penalty & Ponder Loss
                # We disable policy_loss to prevent REINFORCE instability on a fully differentiable graph.
                total_loss = distill_loss + 0.01 * ortho_loss + 0.1 * ponder_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                # Sync gradients back to disk memmap
                student_model.sync_virtual_experts_gradients(lr=1e-3)
                scheduler.step()
                
            loss_val = total_loss.item()
            tokens_trained = x_tensor.numel()
            
            metrics["current_loss"] = round(loss_val, 4)
            metrics["total_tokens_trained"] += tokens_trained
            
            metrics["training_history"].append({
                "timestamp": time.time(),
                "loss": loss_val,
                "source": metrics["last_fetched_url"]
            })
            if len(metrics["training_history"]) > 20:
                metrics["training_history"].pop(0)
                
            if metrics["last_fetched_url"] not in metrics["recent_sources"]:
                metrics["recent_sources"].append(metrics["last_fetched_url"])
                if len(metrics["recent_sources"]) > 5:
                    metrics["recent_sources"].pop(0)
                    
            # Save checkpoint every 10 steps
            if len(metrics["training_history"]) % 10 == 0:
                try:
                    torch.save(student_model.state_dict(), CHECKPOINT_PATH)
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            
        time.sleep(1.5)

# Start background worker only if not suppressed
if os.environ.get("DTSG_NO_TRAIN") != "1":
    distill_thread = threading.Thread(target=background_distillation_loop, daemon=True)
    distill_thread.start()
else:
    print("[Server] Background training suppressed (DTSG_NO_TRAIN=1).")

# ---------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------

@app.get("/status")
def get_status():
    with torch.no_grad():
        emas = student_model.topology_auditor.activation_ema.tolist()
    return JSONResponse({
        "current_loss": metrics["current_loss"],
        "total_tokens_trained": metrics["total_tokens_trained"],
        "last_fetched_url": metrics["last_fetched_url"],
        "training_history": metrics["training_history"],
        "recent_sources": metrics["recent_sources"],
        "topology_activation_emas": [round(v, 4) for v in emas],
        "topology_threshold": student_model.topology_auditor.threshold,
        "reward_baseline": round(metrics["policy_reward_ema"], 4),
        "avg_ponder_steps": metrics["avg_ponder_steps"],
        "active_training_text": metrics["active_training_text"],
        "neuromodulator_level": metrics["neuromodulator_level"]
    })

@app.post("/generate")
async def generate_text(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "Dynamic state graph")
    max_tokens = int(body.get("max_tokens", 35))
    temperature = float(body.get("temperature", 0.7))
    
    if not prompt.strip():
        return JSONResponse({"error": "Prompt cannot be empty"}, status_code=400)
        
    try:
        prompt_ids = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=device)
        
        lookahead_depth = int(body.get("lookahead_depth", 3))
        branching_factor = int(body.get("branching_factor", 2))
        
        with model_lock:
            generated_ids = student_model.generate_mcts(
                prompt_tensor, 
                max_new_tokens=max_tokens, 
                temperature=temperature,
                lookahead_depth=lookahead_depth,
                branching_factor=branching_factor
            )
            
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        return JSONResponse({
            "prompt": prompt,
            "generated_text": generated_text,
            "tokens": generated_ids[0].tolist()
        })
    except Exception as e:
        return JSONResponse({"error": f"Generation failed: {str(e)}"}, status_code=500)

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PhD Spec DTSG: planning Core</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-gradient: linear-gradient(135deg, #020104, #0b071e, #010002);
                --card-bg: rgba(255, 255, 255, 0.03);
                --card-border: rgba(255, 255, 255, 0.06);
                --glow-color: #10b981;
                --text-main: #f5f2ff;
                --text-muted: #9590a8;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                font-family: 'Outfit', sans-serif;
                background: var(--bg-gradient);
                color: var(--text-main);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                overflow-x: hidden;
            }

            header {
                width: 100%;
                max-width: 1200px;
                padding: 2.5rem 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            header h1 {
                font-family: 'Space Grotesk', sans-serif;
                font-weight: 800;
                background: linear-gradient(90deg, #ec4899, #10b981, #6366f1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.3rem;
                letter-spacing: -1px;
            }

            .badge {
                background: rgba(16, 185, 129, 0.15);
                border: 1px solid rgba(16, 185, 129, 0.35);
                color: #a7f3d0;
                padding: 0.5rem 1.2rem;
                border-radius: 50px;
                font-size: 0.85rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.6rem;
                box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
            }

            .badge::before {
                content: '';
                display: inline-block;
                width: 9px;
                height: 9px;
                background-color: #10b981;
                border-radius: 50%;
                animation: pulse 1.4s infinite;
            }

            @keyframes pulse {
                0% { transform: scale(0.85); opacity: 0.5; }
                50% { transform: scale(1.25); opacity: 1; }
                100% { transform: scale(0.85); opacity: 0.5; }
            }

            main {
                width: 100%;
                max-width: 1200px;
                padding: 0 2rem 5rem 2rem;
                display: grid;
                grid-template-columns: 1.15fr 0.85fr;
                gap: 2.5rem;
            }

            @media(max-width: 950px) {
                main { grid-template-columns: 1fr; }
            }

            .glass-card {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                backdrop-filter: blur(20px);
                border-radius: 28px;
                padding: 2.5rem;
                box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.45);
                transition: border-color 0.3s ease;
            }

            .glass-card:hover { border-color: rgba(255, 255, 255, 0.1); }

            h2 {
                font-family: 'Space Grotesk', sans-serif;
                margin-bottom: 1.5rem;
                font-weight: 700;
                font-size: 1.6rem;
                letter-spacing: -0.5px;
                display: flex;
                align-items: center;
                gap: 0.8rem;
            }

            .input-group { margin-bottom: 1.5rem; }

            label {
                display: block;
                font-size: 0.9rem;
                color: var(--text-muted);
                margin-bottom: 0.6rem;
                font-weight: 500;
            }

            textarea, input[type="text"], input[type="number"] {
                width: 100%;
                background: rgba(0, 0, 0, 0.25);
                border: 1px solid var(--card-border);
                border-radius: 14px;
                color: var(--text-main);
                padding: 1.1rem;
                font-family: 'Outfit', sans-serif;
                font-size: 1rem;
                outline: none;
                transition: border-color 0.2s ease, box-shadow 0.2s ease;
            }

            textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
                border-color: rgba(16, 185, 129, 0.45);
                box-shadow: 0 0 12px rgba(16, 185, 129, 0.25);
            }

            textarea { height: 130px; resize: none; }

            .settings-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.2rem;
                margin-bottom: 1.5rem;
            }

            .btn {
                width: 100%;
                background: linear-gradient(135deg, #10b981, #6366f1);
                color: #ffffff;
                border: none;
                border-radius: 14px;
                padding: 1.2rem;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                box-shadow: 0 4px 20px rgba(16, 185, 129, 0.35);
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 25px rgba(16, 185, 129, 0.5);
            }

            .metrics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr 1fr;
                gap: 1.2rem;
                margin-bottom: 2rem;
            }
            .metric-box {
                background: rgba(255, 255, 255, 0.015);
                border: 1px solid var(--card-border);
                border-radius: 18px;
                padding: 1.4rem;
                text-align: center;
            }
            .metric-val {
                font-size: 1.9rem;
                font-weight: 800;
                color: #10b981;
                font-family: 'Space Grotesk', sans-serif;
                margin-top: 0.4rem;
            }
            .metric-lbl {
                font-size: 0.8rem;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .output-box {
                background: rgba(0, 0, 0, 0.35);
                border: 1px solid var(--card-border);
                border-radius: 18px;
                padding: 1.5rem;
                min-height: 150px;
                font-family: 'Outfit', sans-serif;
                line-height: 1.6;
                color: #e5e0fa;
                font-size: 1.1rem;
                margin-top: 1.2rem;
            }

            .log-list {
                list-style-type: none;
                max-height: 200px;
                overflow-y: auto;
                margin-top: 1rem;
                border-top: 1px solid var(--card-border);
                padding-top: 1.2rem;
            }
            .log-item {
                font-size: 0.85rem;
                color: var(--text-muted);
                padding: 0.5rem 0;
                display: flex;
                justify-content: space-between;
                border-bottom: 1px solid rgba(255, 255, 255, 0.015);
            }
            .log-item span.source { color: #10b981; font-weight: 500; }

            .topology-container { margin-top: 2rem; }
            .node-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.9rem;
            }
            .node-name { font-size: 0.9rem; font-weight: 500; }
            .bar-bg {
                background: rgba(255, 255, 255, 0.04);
                height: 10px;
                border-radius: 10px;
                width: 65%;
                overflow: hidden;
            }
            .bar-fill {
                background: linear-gradient(90deg, #10b981, #3b82f6);
                height: 100%;
                width: 0%;
                transition: width 0.4s ease;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>PhD Spec DTSG (Planning-Qwen)</h1>
            <div class="badge">Inline DMCTS & Orthonormal Loss</div>
        </header>

        <main>
            <!-- Generation Interface -->
            <div class="glass-card">
                <h2>📝 Autoregressive Generation</h2>
                <div class="input-group">
                    <label for="prompt">Prompt Input</label>
                    <textarea id="prompt" placeholder="Once upon a time, deep learning representation..."></textarea>
                </div>

                <div class="settings-row">
                    <div>
                        <label for="max-tokens">Tokens Length</label>
                        <input type="number" id="max-tokens" value="35" min="5" max="150">
                    </div>
                    <div>
                        <label for="temperature">Creativity (Temp)</label>
                        <input type="number" id="temperature" value="0.7" step="0.1" min="0.1" max="2.0">
                    </div>
                </div>

                <div class="settings-row">
                    <div>
                        <label for="lookahead-depth">MCTS Depth</label>
                        <input type="number" id="lookahead-depth" value="3" min="1" max="5">
                    </div>
                    <div>
                        <label for="branching-factor">MCTS Branching</label>
                        <input type="number" id="branching-factor" value="2" min="1" max="4">
                    </div>
                </div>

                <button class="btn" id="generate-btn">Execute Graph</button>

                <h3 style="margin-top: 2.2rem; font-size: 1rem; font-weight: 600;">planning Completion</h3>
                <div class="output-box" id="output">Generated outputs using bootstrapped learning will show here...</div>
            </div>

            <!-- Distillation Logs -->
            <div class="glass-card">
                <h2>⚡ Live Optimization Metrics</h2>
                
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-lbl">Total Loss (CE + RL)</div>
                        <div class="metric-val" id="current-loss">0.0000</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-lbl">RL Reward Baseline</div>
                        <div class="metric-val" id="reward-val">0.0000</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-lbl">Ponder Steps</div>
                        <div class="metric-val" id="ponder-steps-val">0.00</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-lbl">ACh (Surprise)</div>
                        <div class="metric-val" id="neuromodulator-val">1.000</div>
                    </div>
                </div>

                <div class="input-group">
                    <label>Active Dataset Corpus</label>
                    <input type="text" id="live-source" readonly value="Syncing...">
                </div>

                <div class="input-group">
                    <label>Real-Time Training Text Stream</label>
                    <div id="live-text" style="font-family: monospace; font-size: 0.85rem; padding: 0.8rem; height: 100px; overflow-y: auto; background: rgba(0,0,0,0.2); border: 1px solid var(--card-border); border-radius: 10px; color: #a7f3d0; line-height: 1.4; white-space: pre-wrap; word-break: break-word;">Syncing active stream...</div>
                </div>

                <div class="topology-container">
                    <label>Node Activations EMA (Bypass at 5.0)</label>
                    <div style="margin-top: 1rem;">
                        <div class="node-row">
                            <span class="node-name">Enzyme 0</span>
                            <div class="bar-bg"><div class="bar-fill" id="ema-0"></div></div>
                        </div>
                        <div class="node-row">
                            <span class="node-name">Enzyme 1</span>
                            <div class="bar-bg"><div class="bar-fill" id="ema-1"></div></div>
                        </div>
                        <div class="node-row">
                            <span class="node-name">Enzyme 2</span>
                            <div class="bar-bg"><div class="bar-fill" id="ema-2"></div></div>
                        </div>
                    </div>
                </div>

                <h3 style="margin-top: 1.8rem; font-size: 1rem; font-weight: 600;">Optimization History Logs</h3>
                <ul class="log-list" id="log-list">
                    <li style="padding: 0.5rem; text-align: center; color: var(--text-muted);">Syncing logs...</li>
                </ul>
            </div>
        </main>

        <script>
            const generateBtn = document.getElementById("generate-btn");
            const promptInput = document.getElementById("prompt");
            const maxTokensInput = document.getElementById("max-tokens");
            const tempInput = document.getElementById("temperature");
            const lookaheadDepthInput = document.getElementById("lookahead-depth");
            const branchingFactorInput = document.getElementById("branching-factor");
            const outputBox = document.getElementById("output");

            const currentLossEl = document.getElementById("current-loss");
            const rewardValEl = document.getElementById("reward-val");
            const ponderStepsValEl = document.getElementById("ponder-steps-val");
            const neuromodulatorValEl = document.getElementById("neuromodulator-val");
            const liveSourceEl = document.getElementById("live-source");
            const liveTextEl = document.getElementById("live-text");
            const logListEl = document.getElementById("log-list");

            async function updateStatus() {
                try {
                    const res = await fetch("/status");
                    const data = await res.json();

                    currentLossEl.textContent = data.current_loss;
                    rewardValEl.textContent = data.reward_baseline;
                    ponderStepsValEl.textContent = data.avg_ponder_steps;
                    neuromodulatorValEl.textContent = data.neuromodulator_level.toFixed(3);
                    liveSourceEl.value = data.last_fetched_url;
                    liveTextEl.textContent = data.active_training_text || "None";

                    const threshold = data.topology_threshold;
                    data.topology_activation_emas.forEach((ema, idx) => {
                        const percent = Math.min((ema / threshold) * 100, 100);
                        const fillEl = document.getElementById(`ema-${idx}`);
                        if (fillEl) {
                            fillEl.style.width = `${percent}%`;
                            if (percent > 85) {
                                fillEl.style.background = "linear-gradient(90deg, #ef4444, #f43f5e)";
                            } else {
                                fillEl.style.background = "linear-gradient(90deg, #10b981, #3b82f6)";
                            }
                        }
                    });

                    if (data.training_history && data.training_history.length > 0) {
                        logListEl.innerHTML = "";
                        data.training_history.slice().reverse().forEach(item => {
                            const dateStr = new Date(item.timestamp * 1000).toLocaleTimeString();
                            const li = document.createElement("li");
                            li.className = "log-item";
                            li.innerHTML = `
                                <span>[${dateStr}] Loss: <strong style="color:#3b82f6">${item.loss.toFixed(4)}</strong></span>
                                <span class="source">${item.source}</span>
                            `;
                            logListEl.appendChild(li);
                        });
                    }
                } catch (err) {
                    console.error("Error updating status:", err);
                }
            }

            setInterval(updateStatus, 2000);
            updateStatus();

            generateBtn.addEventListener("click", async () => {
                const prompt = promptInput.value;
                if (!prompt.trim()) {
                    alert("Please type a seed prompt.");
                    return;
                }

                generateBtn.disabled = true;
                generateBtn.textContent = "Synthesizing...";
                outputBox.textContent = "Processing planning loop...";

                try {
                    const response = await fetch("/generate", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            prompt: prompt,
                            max_tokens: parseInt(maxTokensInput.value) || 35,
                            temperature: parseFloat(tempInput.value) || 0.7,
                            lookahead_depth: parseInt(lookaheadDepthInput.value) || 3,
                            branching_factor: parseInt(branchingFactorInput.value) || 2
                        })
                    });
                    const data = await response.json();
                    if (data.error) {
                        outputBox.textContent = "Error: " + data.error;
                    } else {
                        outputBox.textContent = data.generated_text;
                    }
                } catch (err) {
                    outputBox.textContent = "Failed to compile: " + err.message;
                } finally {
                    generateBtn.disabled = false;
                    generateBtn.textContent = "Execute Graph";
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
