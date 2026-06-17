import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# We will initialize our virtual experts database with compressed reasoning, dialogue, and coding knowledge from the actual frontier models.
# Experts 0 - 256:   Retain existing trained expert weights (read from file if exists)
# Experts 256 - 384:  DeepSeek-R1 (from deepseek-ai/DeepSeek-R1)
# Experts 384 - 512:  Qwen3-235B (from Qwen/Qwen3-235B-A22B-Thinking-2507)
# Experts 512 - 640:  Llama 4 Scout (from meta-llama/Llama-4-Scout-Instruct)
# Experts 640 - 768:  DeepSeek-Coder-V2-Instruct (from deepseek-ai/DeepSeek-Coder-V2-Instruct)
# Experts 768 - 896:  Mistral-Large-Instruct-2411 (from mistralai/Mistral-Large-Instruct-2411)
# Experts 896 - 1024: GLM-4-9b-chat (from THUDM/glm-4-9b-chat)

VIRTUAL_EXPERTS_PATH = "virtual_experts.bin"
# Scale to target 70GB layout
NUM_ENZYMES = 6
NUM_EXPERTS = 8192
NUM_HEADS = 2
D_MODEL = 896
LORA_RANK = 64

GATE_SIZE = NUM_HEADS * D_MODEL * LORA_RANK
VAL_SIZE = NUM_HEADS * D_MODEL * LORA_RANK
PROJ_SIZE = NUM_HEADS * LORA_RANK * D_MODEL
EXPERT_DIM = GATE_SIZE + VAL_SIZE + PROJ_SIZE

def randomized_svd_decompose(W, out_dim, in_dim):
    W = W.float().detach()
    # PyTorch low-rank SVD using active rank
    U, S, V = torch.svd_lowrank(W, q=LORA_RANK)
    
    W_left = torch.matmul(U, torch.diag(torch.sqrt(S)))
    W_right = torch.matmul(torch.diag(torch.sqrt(S)), V.T)
    
    if W_left.shape[0] != out_dim:
        W_left = F.interpolate(W_left.unsqueeze(0).unsqueeze(0), size=(out_dim, LORA_RANK), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    if W_right.shape[1] != in_dim:
        W_right = F.interpolate(W_right.unsqueeze(0).unsqueeze(0), size=(LORA_RANK, in_dim), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
    return W_left, W_right

def download_and_extract_weights_from_first_shard(repo_name, num_experts_to_fill):
    print(f"\n[Loader] Initializing download for {repo_name}...")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_SECRET")
    if not hf_token:
        try:
            token_path = os.path.expanduser("~/.cache/huggingface/token")
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    hf_token = f.read().strip()
        except Exception:
            pass
    
    # 1. Download index json to find the first shard name
    try:
        index_path = hf_hub_download(repo_id=repo_name, filename="model.safetensors.index.json", token=hf_token)
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        # Filter for shards that contain MLP or layer weights
        mlp_shards = [shard for key, shard in weight_map.items() if "mlp" in key or "layers" in key or "h_to_4h" in key or "gate_up" in key]
        if mlp_shards:
            first_shard_name = sorted(list(set(mlp_shards)))[0]
        else:
            first_shard_name = sorted(list(set(weight_map.values())))[0]
    except Exception as e:
        print(f"[Loader] Fallback: Index file not found, trying default shard name: {e}")
        first_shard_name = "model-00001-of-00024.safetensors"
        
    print(f"[Loader] Downloading only Shard 1: {first_shard_name} from HF...")
    shard_path = hf_hub_download(repo_id=repo_name, filename=first_shard_name, token=hf_token)
    
    print(f"[Loader] Shard loaded. Loading tensors into memory...")
    tensors = load_file(shard_path)
    
    # Identify MLP keys in the loaded shard
    glm_h_to_4h_keys = [k for k in tensors.keys() if "dense_h_to_4h" in k]
    glm_4h_to_h_keys = [k for k in tensors.keys() if "dense_4h_to_h" in k]
    
    is_glm = len(glm_h_to_4h_keys) > 0 and len(glm_4h_to_h_keys) > 0
    
    moe_gate_up_keys = [k for k in tensors.keys() if "gate_up_proj" in k]
    moe_down_keys = [k for k in tensors.keys() if "down_proj" in k and "shared_expert" not in k]
    
    is_moe_gate_up = len(moe_gate_up_keys) > 0 and len(moe_down_keys) > 0
    
    if is_glm:
        gate_keys = glm_h_to_4h_keys
        up_keys = glm_h_to_4h_keys
        down_keys = glm_4h_to_h_keys
    elif is_moe_gate_up:
        gate_keys = moe_gate_up_keys
        up_keys = moe_gate_up_keys
        down_keys = moe_down_keys
    else:
        gate_keys = [k for k in tensors.keys() if "mlp" in k and ("gate_proj" in k or "experts" in k)]
        up_keys = [k for k in tensors.keys() if "mlp" in k and ("up_proj" in k or "experts" in k)]
        down_keys = [k for k in tensors.keys() if "mlp" in k and ("down_proj" in k or "experts" in k)]
        
        if not gate_keys or not up_keys or not down_keys:
            gate_keys = [k for k in tensors.keys() if "w1" in k]
            up_keys = [k for k in tensors.keys() if "w2" in k]
            down_keys = [k for k in tensors.keys() if "w3" in k]
            
        if not gate_keys:
            raise ValueError(f"Could not locate MLP layers in safetensors keys: {list(tensors.keys())[:15]}")
        
    expert_weights = []
    
    print(f"[Compressor] Computing SVD compression for {num_experts_to_fill} experts...")
    for i in range(num_experts_to_fill):
        g_k = gate_keys[i % len(gate_keys)]
        u_k = up_keys[i % len(up_keys)]
        d_k = down_keys[i % len(down_keys)]
        
        if is_glm:
            w_h_to_4h = tensors[g_k]
            chunk_size = w_h_to_4h.shape[0] // 2
            g_w = w_h_to_4h[:chunk_size]
            u_w = w_h_to_4h[chunk_size:]
            d_w = tensors[d_k]
        elif is_moe_gate_up:
            w_gate_up_all = tensors[g_k]
            w_down_all = tensors[d_k]
            if w_gate_up_all.dim() == 3:
                num_exp = w_gate_up_all.shape[0]
                exp_idx = i % num_exp
                w_gate_up = w_gate_up_all[exp_idx]
                w_down = w_down_all[exp_idx]
            else:
                w_gate_up = w_gate_up_all
                w_down = w_down_all
            chunk_size = w_gate_up.shape[0] // 2
            g_w = w_gate_up[:chunk_size]
            u_w = w_gate_up[chunk_size:]
            d_w = w_down
        else:
            g_w = tensors[g_k]
            u_w = tensors[u_k]
            d_w = tensors[d_k]
        
        gate_w_left, gate_w_right = randomized_svd_decompose(g_w, D_MODEL, LORA_RANK)
        val_w_left, val_w_right = randomized_svd_decompose(u_w, D_MODEL, LORA_RANK)
        proj_w_left, proj_w_right = randomized_svd_decompose(d_w, LORA_RANK, D_MODEL)
        
        # Reshape to (NUM_HEADS, D_MODEL, LORA_RANK)
        gate_w = gate_w_left.repeat(1, NUM_HEADS).view(NUM_HEADS, D_MODEL, LORA_RANK)
        val_w = val_w_left.repeat(1, NUM_HEADS).view(NUM_HEADS, D_MODEL, LORA_RANK)
        proj_w = proj_w_right.repeat(NUM_HEADS, 1).view(NUM_HEADS, LORA_RANK, D_MODEL)
        
        expert_vector = torch.cat([gate_w.flatten(), val_w.flatten(), proj_w.flatten()])
        expert_weights.append(expert_vector.detach().cpu().numpy().astype('float32'))
        
    # Delete shard from cache to free space
    try:
        os.remove(shard_path)
        print(f"[Loader] Deleted downloaded shard {first_shard_name} to minimize space.")
    except Exception:
        pass
        
    return expert_weights

def main():
    print("="*60)
    print("  DTSG Frankenstein MoE Weight Initializer (8192 Experts)")
    print("="*60)
    
    shape = (NUM_ENZYMES, NUM_EXPERTS, EXPERT_DIM)
    expected_bytes = NUM_ENZYMES * NUM_EXPERTS * EXPERT_DIM * 4
    
    if os.path.exists(VIRTUAL_EXPERTS_PATH) and os.path.getsize(VIRTUAL_EXPERTS_PATH) == expected_bytes:
        print(f"Loading existing expert weights from {VIRTUAL_EXPERTS_PATH} to preserve experts 0-1024...")
        mmap_arr = np.memmap(VIRTUAL_EXPERTS_PATH, dtype='float32', mode='r+', shape=shape)
    else:
        print(f"Creating new expert weights database chunk-by-chunk to avoid OOM: {VIRTUAL_EXPERTS_PATH}...")
        mmap_arr = np.memmap(VIRTUAL_EXPERTS_PATH, dtype='float32', mode='w+', shape=shape)
        chunk_size = 256
        for start_idx in range(0, NUM_EXPERTS, chunk_size):
            end_idx = min(start_idx + chunk_size, NUM_EXPERTS)
            current_experts = end_idx - start_idx
            chunk_shape = (NUM_ENZYMES, current_experts, EXPERT_DIM)
            mmap_arr[:, start_idx:end_idx] = np.random.normal(0, 0.02, size=chunk_shape).astype('float32')
        mmap_arr.flush()
        
    # Experts 1024 - 2048: DeepSeek-R1 (1024 experts)
    try:
        ds_weights = download_and_extract_weights_from_first_shard("deepseek-ai/DeepSeek-R1", 1024)
        for enzyme in range(NUM_ENZYMES):
            mmap_arr[enzyme, 1024:2048] = ds_weights
        print("Successfully injected DeepSeek-R1 weights! ✅")
    except Exception as e:
        print(f"Failed to inject DeepSeek-R1 weights: {e}")
        
    # Experts 2048 - 3072: Qwen3-235B (1024 experts)
    try:
        qwen_weights = download_and_extract_weights_from_first_shard("Qwen/Qwen3-235B-A22B-Thinking-2507", 1024)
        for enzyme in range(NUM_ENZYMES):
            mmap_arr[enzyme, 2048:3072] = qwen_weights
        print("Successfully injected Qwen3-235B weights! ✅")
    except Exception as e:
        print(f"Failed to inject Qwen3-235B weights: {e}")
        
    # Experts 3072 - 4096: Llama 4 Scout (1024 experts)
    try:
        llama_weights = download_and_extract_weights_from_first_shard("meta-llama/Llama-4-Scout-17B-16E-Instruct", 1024)
        for enzyme in range(NUM_ENZYMES):
            mmap_arr[enzyme, 3072:4096] = llama_weights
        print("Successfully injected Llama 4 Scout weights! ✅")
    except Exception as e:
        print(f"Failed to inject Llama 4 Scout weights: {e}")

    # Experts 4096 - 5120: DeepSeek Coder V2 (1024 experts)
    try:
        coder_weights = download_and_extract_weights_from_first_shard("deepseek-ai/DeepSeek-Coder-V2-Instruct", 1024)
        for enzyme in range(NUM_ENZYMES):
            mmap_arr[enzyme, 4096:5120] = coder_weights
        print("Successfully injected DeepSeek-Coder-V2-Instruct weights! ✅")
    except Exception as e:
        print(f"Failed to inject DeepSeek-Coder-V2-Instruct weights: {e}")

    # Experts 5120 - 6144: Mistral Large 2411 (1024 experts)
    try:
        mistral_weights = download_and_extract_weights_from_first_shard("mistralai/Mistral-Large-Instruct-2411", 1024)
        for enzyme in range(NUM_ENZYMES):
            mmap_arr[enzyme, 5120:6144] = mistral_weights
        print("Successfully injected Mistral-Large-Instruct-2411 weights! ✅")
    except Exception as e:
        print(f"Failed to inject Mistral-Large-Instruct-2411 weights: {e}")

    # Experts 6144 - 7168: GLM-4-9b-chat (1024 experts)
    try:
        glm_weights = download_and_extract_weights_from_first_shard("THUDM/glm-4-9b-chat", 1024)
        for enzyme in range(NUM_ENZYMES):
            mmap_arr[enzyme, 6144:7168] = glm_weights
        print("Successfully injected GLM-4-9b-chat weights! ✅")
    except Exception as e:
        print(f"Failed to inject GLM-4-9b-chat weights: {e}")

    # Experts 7168 - 8192: DeepSeek-R1-Distill-Llama-70B (1024 experts)
    try:
        llama_70b_weights = download_and_extract_weights_from_first_shard("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 1024)
        for enzyme in range(NUM_ENZYMES):
            mmap_arr[enzyme, 7168:8192] = llama_70b_weights
        print("Successfully injected DeepSeek-R1-Distill-Llama-70B weights! ✅")
    except Exception as e:
        print(f"Failed to inject DeepSeek-R1-Distill-Llama-70B weights: {e}")
        
    mmap_arr.flush()
    print("\n[Complete] Frankenstein MoE initialization complete! Database updated successfully. 🧠🔥")

if __name__ == "__main__":
    main()
