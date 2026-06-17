import modal
import os

app = modal.App("dtsg-training")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "transformers", "numpy",
        "bitsandbytes",           # 8-bit quantised teacher loading
        "accelerate",             # required by bitsandbytes device_map
        "datasets",               # FineWeb-Edu / SlimOrca / OpenHermes streaming
        "huggingface_hub",
    )
    .add_local_file("agent_model.py", "/root/agent_model.py")
)

volume = modal.Volume.from_name("dtsg-training-vol", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="A100-80GB",
    timeout=3600,        # 60 min limit
)
def train(num_steps: int = 3000):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import time
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from agent_model import DTSGModel, SwappableVirtualExperts
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"[Modal] GPU: {gpu_name} | VRAM: {gpu_mem_gb:.1f} GB")

    # Use DeepSeek-R1-Distill-Qwen-32B for advanced R1 chain-of-thought reasoning
    TEACHER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    print(f"[Modal] Selected teacher: {TEACHER_NAME}")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from agent_model import DTSGModel, SwappableVirtualExperts
    import random, time

    # Explicitly pull the HF token from the Modal secret injection or local fallback
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_SECRET") or None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("[Modal] Authenticating HuggingFace Hub with HF_TOKEN ✅")
    else:
        print("[Modal] Warning: HF_TOKEN not found in environment secrets ⚠️")
        
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME, token=hf_token)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NF4 is higher quality than fp4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,      # double quant saves another ~0.4 bpw
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print(f"[Modal] Teacher loaded (4-bit NF4). VRAM used: "
          f"{torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Student model ──────────────────────────────────────────────────────
    vol_experts_path    = "/vol/virtual_experts.bin"
    vol_checkpoint_path = "/vol/dtsg_checkpoint.pt"

    student_model = DTSGModel(
        teacher_model_name="Qwen/Qwen2.5-0.5B",
        num_enzymes=6,
        max_loops=8,
        teacher_model=None,
        virtual_experts_path=vol_experts_path,
    ).to(device)

    if os.path.exists(vol_checkpoint_path):
        try:
            print("[Modal] Loading checkpoint from volume...")
            checkpoint_state = torch.load(vol_checkpoint_path, map_location='cpu')
            # Remove any keys with size mismatches or teacher keys to allow smooth resume
            model_state = student_model.state_dict()
            matched_state = {}
            for k, v in checkpoint_state.items():
                # Strip legacy teacher parameters from checkpoint to save GPU memory
                if any(x in k for x in ["teacher_model", "embedding", "lm_head", "_teacher_model", "_embedding", "_lm_head"]):
                    continue
                if k in model_state:
                    if model_state[k].shape == v.shape:
                        matched_state[k] = v.to(device)
                    elif k == "router.gating.weight":
                        new_v = torch.zeros(6, 896, dtype=v.dtype)
                        new_v[:3] = v
                        new_v[3:] = v
                        matched_state[k] = new_v.to(device)
                    elif k == "router.gating.bias":
                        new_v = torch.zeros(6, dtype=v.dtype)
                        new_v[:3] = v
                        new_v[3:] = v
                        matched_state[k] = new_v.to(device)
                    elif k == "topology_auditor.activation_ema":
                        new_v = torch.zeros(6, dtype=v.dtype)
                        new_v[:3] = v
                        new_v[3:] = v
                        matched_state[k] = new_v.to(device)
                    else:
                        print(f"[Modal] Skipping key {k} due to shape/size change or exclusion")
            student_model.load_state_dict(matched_state, strict=False)
            print("[Modal] Checkpoint loaded (matched keys successfully resumed) ✅")
            del checkpoint_state
            del matched_state
            import gc; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"[Modal] Checkpoint load failed (starting fresh): {e}")

    hypernetwork_params = []
    gating_params       = []
    projection_params   = []

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
        {"params": hypernetwork_params, "lr": 1e-3, "weight_decay": 0.01},
        {"params": gating_params,       "lr": 5e-4, "weight_decay": 0.01},
        {"params": projection_params,   "lr": 1e-4, "weight_decay": 0.01},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=1e-5)

    # ── Dataset: interleaved high-quality streams ─────────────────────────────
    from datasets import load_dataset, interleave_datasets
    print("[Modal] Loading datasets...")

    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                           split="train", streaming=True, trust_remote_code=True)
    slimorca = load_dataset("Open-Orca/SlimOrca",
                            split="train", streaming=True, trust_remote_code=True)
    openhermes = load_dataset("teknium/OpenHermes-2.5",
                              split="train", streaming=True, trust_remote_code=True)
    gsm8k = load_dataset("openai/gsm8k", "main",
                         split="train", streaming=True, trust_remote_code=True)

    def _text(row, ds_name):
        """Extract raw text from any dataset row format."""
        t_val = ""
        if "text" in row:
            t_val = row["text"]
        elif "conversations" in row:
            t_val = " ".join(t.get("value","") for t in row["conversations"] if isinstance(t, dict))
        elif "question" in row:
            t_val = str(row["question"]) + " " + str(row.get("answer",""))
        else:
            t_val = str(row)
        if isinstance(t_val, str):
            return t_val
        return str(t_val)

    # Weighted interleave: 50% FineWeb, 20% SlimOrca, 20% OpenHermes, 10% GSM8K
    mixed = interleave_datasets(
        [fineweb, slimorca, openhermes, gsm8k],
        probabilities=[0.5, 0.2, 0.2, 0.1],
        seed=42,
    )
    dataset_iter = iter(mixed)
    DS_NAMES = ["fineweb", "slimorca", "openhermes", "gsm8k"]
    print("[Modal] Datasets ready ✅")
        
    alpha, beta, gamma = 0.4, 0.4, 0.2
    temperature = 2.0
    seq_len = 256         # scale down to fit VRAM
    batch_size = 8        # scale down to fit VRAM
    reward_momentum = 0.95
    policy_reward_ema = -5.0
    start_wall = time.time()
    WALL_LIMIT  = 3300    # stop at 55 min to stay under 60-min container cap
    
    def info_nce_loss(features, temp=0.1):
        features = F.normalize(features, dim=-1)
        similarity_matrix = torch.matmul(features, features.T)
        exp_sim = torch.exp(similarity_matrix / temp)
        mask = ~torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
        exp_sim_masked = exp_sim * mask.float()
        sum_negatives = torch.sum(exp_sim_masked, dim=-1)
        loss = -torch.log(torch.clamp(torch.diag(exp_sim) / (torch.diag(exp_sim) + sum_negatives), min=1e-10))
        return loss.mean()
        
    print(f"[Modal] Starting distillation: {num_steps} steps...")
    for step in range(1, num_steps + 1):
        # Hard wall-clock limit to avoid overspend
        if time.time() - start_wall > WALL_LIMIT:
            print(f"[Modal] Wall-clock limit reached at step {step}. Stopping.")
            break

        start_time = time.time()

        # ── Build batch from mixed dataset stream ────────────────────────────
        x_batches, y_batches = [], []
        while len(x_batches) < batch_size:
            try:
                row = next(dataset_iter)
                text = _text(row, "mixed")
            except StopIteration:
                dataset_iter = iter(mixed)
                continue
            except Exception:
                continue
            tokens = tokenizer.encode(text)
            if len(tokens) > seq_len + 1:
                x_batches.append(tokens[:seq_len])
                y_batches.append(tokens[1:seq_len + 1])
                
        # GRPO Group Rollout Parameter
        G = 4
        
        optimizer.zero_grad()
        
        # Build batch data
        x_tensor = torch.tensor(x_batches, device=device)
        y_tensor = torch.tensor(y_batches, device=device)
        
        # AUTOCAST mixed precision training context (AMP)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            group_logits = []
            group_s_final = []
            group_policy_log_probs = []
            group_ponder_steps = []
            group_entropy = []
            
            # 1. Sample G sibling rollouts
            for g in range(G):
                student_logits, s_final, policy_log_probs, s_contrastive, ponder_loss, halt_probs, ach_val, _ = student_model(x_tensor)
                
                # Calculate routing entropy
                student_probs = F.softmax(student_logits, dim=-1)
                entropy = -torch.sum(student_probs * torch.log(torch.clamp(student_probs, min=1e-10)), dim=-1).mean()
                
                group_logits.append(student_logits)
                group_s_final.append(s_final)
                group_policy_log_probs.append(policy_log_probs)
                group_ponder_steps.append(len(halt_probs))
                group_entropy.append(entropy)
                
            # 2. Compute Rewards and Group-Relative Advantages
            # Consistency reward: Cosine similarity of s_final to the group mean
            stacked_s = torch.stack(group_s_final, dim=0) # (G, batch, seq, d_model)
            mean_s = stacked_s.mean(dim=0, keepdim=True) # (1, batch, seq, d_model)
            
            rewards = []
            for g in range(G):
                cos_sim = F.cosine_similarity(group_s_final[g], mean_s.squeeze(0), dim=-1).mean()
                
                # Reward formula: Consensus + Entropy Exploration - Loop Efficiency Penalty
                reward_val = cos_sim.item() + 0.1 * group_entropy[g].item() - 0.05 * group_ponder_steps[g]
                rewards.append(reward_val)
                
            # Normalize group rewards to get advantages
            rewards_tensor = torch.tensor(rewards, device=device)
            r_mean = rewards_tensor.mean()
            r_std = rewards_tensor.std().clamp(min=1e-8)
            advantages = (rewards_tensor - r_mean) / r_std
            
            # 3. Calculate Policy Loss and Distillation loss
            policy_loss = 0.0
            distill_loss = 0.0
            
            for g in range(G):
                # Cross-entropy base loss
                ce_loss = F.cross_entropy(group_logits[g].view(-1, student_model.vocab_size), y_tensor.view(-1))
                distill_loss = distill_loss + ce_loss
                
                # Policy gradient updates modulated by advantage
                if group_policy_log_probs[g]:
                    stacked_log_probs = torch.stack(group_policy_log_probs[g], dim=0)
                    policy_loss = policy_loss - torch.mean(stacked_log_probs) * advantages[g]
                    
            distill_loss = distill_loss / G
            policy_loss = policy_loss / G
            ortho_loss = student_model.get_orthonormal_penalty()
            
            total_loss = distill_loss + 0.1 * policy_loss + 0.01 * ortho_loss
            
        total_loss.backward()
        nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        step_time = time.time() - start_time
        
        if step % 20 == 0 or step == 1:
            print(f"[Step {step}/{num_steps}] Loss: {total_loss.item():.4f} | R-Mean: {r_mean.item():.2f} | ACh: {ach_val:.2f} | Mean Loops: {sum(group_ponder_steps)/G:.1f} | Time/step: {step_time:.3f}s")
            
        if step % 500 == 0:
            print("Saving checkpoint to volume...")
            torch.save(student_model.state_dict(), vol_checkpoint_path)
            volume.commit()
            
    print("Training finished! Saving final states...")
    torch.save(student_model.state_dict(), vol_checkpoint_path)
    volume.commit()
    print("Volume sync finished successfully.")

@app.local_entrypoint()
def main():
    print("Launching training task on Modal cloud...")
    train.remote()
    print("Training finished. Run 'modal volume get dtsg-training-vol dtsg_checkpoint.pt dtsg_checkpoint.pt' locally to retrieve the checkpoint.")
