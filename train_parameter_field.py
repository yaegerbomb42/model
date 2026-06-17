import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("/Users/yaeger/Desktop/model")
from dtsg_model import DTSGModel

VIRTUAL_EXPERTS_PATH = "/Users/yaeger/Desktop/model/virtual_experts.bin"
CHECKPOINT_PATH = "/Users/yaeger/Desktop/model/dtsg_checkpoint.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 1. Load target expert weights from virtual_experts.bin
    NUM_ENZYMES = 6
    NUM_EXPERTS = 1024
    EXPERT_DIM = 86016 # 2 * 896 * 16 * 3
    shape = (NUM_ENZYMES, NUM_EXPERTS, EXPERT_DIM)
    
    print(f"Loading target weights from {VIRTUAL_EXPERTS_PATH}...")
    mmap_arr = np.memmap(VIRTUAL_EXPERTS_PATH, dtype='float32', mode='r', shape=shape)
    # Convert only to float32 tensor
    target_weights = torch.from_numpy(np.array(mmap_arr)).to(device)
    print("Target weights loaded. Shape:", target_weights.shape)
    
    # 2. Instantiate the new model (which contains the parameter generators)
    print("Loading base seed model and initializing DTSGModel...")
    student_model = DTSGModel(teacher_model_name="Qwen/Qwen2.5-0.5B", num_enzymes=NUM_ENZYMES, max_loops=8, teacher_model=None).to(device)
    
    # Load existing trained checkpoints (to preserve trained embeddings and router gating weights)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming trained checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model_state = student_model.state_dict()
        matched_state = {}
        for k, v in checkpoint.items():
            if not k.startswith("teacher_model") and k in model_state:
                # Map 3-enzyme sizes to 6-enzyme sizes if needed
                if k == "router.gating.weight" and v.shape != model_state[k].shape:
                    new_v = torch.zeros(6, 896, dtype=v.dtype)
                    new_v[:3] = v
                    new_v[3:] = v
                    matched_state[k] = new_v
                elif k == "router.gating.bias" and v.shape != model_state[k].shape:
                    new_v = torch.zeros(6, dtype=v.dtype)
                    new_v[:3] = v
                    new_v[3:] = v
                    matched_state[k] = new_v
                elif k == "topology_auditor.activation_ema" and v.shape != model_state[k].shape:
                    new_v = torch.zeros(6, dtype=v.dtype)
                    new_v[:3] = v
                    new_v[3:] = v
                    matched_state[k] = new_v
                elif model_state[k].shape == v.shape:
                    matched_state[k] = v
        student_model.load_state_dict(matched_state, strict=False)
        print("Checkpoint successfully loaded and adapted! ✅")

    # Move model to target device and datatype
    if device.type == "mps":
        student_model = student_model.to(torch.bfloat16)
        target_weights = target_weights.to(torch.bfloat16)
    else:
        student_model = student_model.float()
        target_weights = target_weights.float()

    # 3. Train each enzyme's parameter_generator to reconstruct its virtual experts
    print("\n" + "="*60)
    print("  Pre-training Continuous Parameter Generators (HyperNetwork compression)")
    print("="*60)
    
    # Gather parameter generator parameters to optimize
    generators_params = []
    for enzyme in student_model.enzymes:
        generators_params.append({"params": enzyme.parameter_generator.parameters(), "lr": 1e-3})
        generators_params.append({"params": [enzyme.anchor_coords], "lr": 5e-3})
        
    optimizer = torch.optim.AdamW(generators_params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-5)
    
    # Pre-train loops
    steps = 10000
    pbar = tqdm(range(steps), desc="Compression training")
    for step in pbar:
        optimizer.zero_grad()
        
        total_loss = 0.0
        for idx, enzyme in enumerate(student_model.enzymes):
            # Map coordinate embeddings directly to parameters
            generated_weights = enzyme.parameter_generator(enzyme.anchor_coords)  # (1024, EXPERT_DIM)
            loss = F.mse_loss(generated_weights, target_weights[idx])
            total_loss += loss
            
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0:
            pbar.set_postfix({"Loss": f"{total_loss.item():.6f}"})
            
    print("\nCompression complete! Saving updated model checkpoint...")
    # Move model back to CPU float32 before saving
    student_model = student_model.cpu().float()
    torch.save(student_model.state_dict(), CHECKPOINT_PATH)
    print(f"Checkpoint successfully saved to {CHECKPOINT_PATH}! ✅")
    print("You can now safely delete the 19.3 GB virtual_experts.bin file to free disk space.")

if __name__ == "__main__":
    main()
