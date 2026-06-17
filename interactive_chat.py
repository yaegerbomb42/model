import torch
import sys
import os
from transformers import AutoTokenizer
from agent_model import DTSGModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("="*60)
    print("  DTSG Interactive Frankenstein MoE Chat Interface")
    print("="*60)
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    print("\nLoading model structure...")
    student_model = DTSGModel(teacher_model_name="Qwen/Qwen2.5-0.5B", num_enzymes=6, max_loops=8, teacher_model=None).to(device)
    
    checkpoint_path = "/Users/yaeger/Desktop/model/dtsg_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        student_state_dict = {}
        model_state = student_model.state_dict()
        for k, v in checkpoint.items():
            if not k.startswith("teacher_model"):
                if k in model_state and model_state[k].shape == v.shape:
                    student_state_dict[k] = v
        student_model.load_state_dict(student_state_dict, strict=False)
        print("Checkpoint loaded successfully!")
    else:
        print("No checkpoint found. Running with base seed weights.")
        
    if device.type == "mps":
        student_model = student_model.to(torch.bfloat16)
        
    student_model.eval()
    
    # Model partition labels for diagnostic printing
    partitions = [
        "Trained Checkpoint (0-256)",
        "DeepSeek-R1 (256-352)",
        "Qwen3-235B (352-448)",
        "Llama 4 Scout (448-544)",
        "DeepSeek Coder V2 (544-640)",
        "Mistral Large 2411 (640-736)",
        "GLM-4-9b-chat (736-832)",
        "Qwen2-VL-7B-Instruct (832-928)",
        "DeepSeek-R1-Distill-Llama-70B (928-1024)"
    ]
    
    def get_partition_name(idx):
        if idx < 256: return partitions[0]
        elif idx < 352: return partitions[1]
        elif idx < 448: return partitions[2]
        elif idx < 544: return partitions[3]
        elif idx < 640: return partitions[4]
        elif idx < 736: return partitions[5]
        elif idx < 832: return partitions[6]
        elif idx < 928: return partitions[7]
        else: return partitions[8]

    print("\nModel is ready for interaction! 🧠🔥")
    print("Type your prompt and press Enter. Type 'exit' to quit.")
    print("-" * 60)
    
    while True:
        try:
            prompt = input("\nPrompt > ")
            if not prompt.strip():
                continue
            if prompt.strip().lower() == "exit":
                break
                
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            print("Generating (Standard Routing)...")
            with torch.no_grad():
                # Let's perform a single step generation to trace routing paths for diagnostics
                # and then generate the remaining output.
                logits, s, policy_log_probs, _, _, halt_probs, mean_ach, _ = student_model(inputs.input_ids)
                
                # Run full generate
                output_tokens = student_model.generate(inputs.input_ids, max_new_tokens=25)
                
            decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            print(f"\nGenerated Output:\n{decoded}")
            print("-" * 40)
            print("Diagnostics:")
            print(f"  Average Surprise Level (ACh): {mean_ach:.2f}")
            print(f"  Ponder Loops Taken: {len(halt_probs)}")
            
        except KeyboardInterrupt:
            print("\nExiting interactive interface.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
