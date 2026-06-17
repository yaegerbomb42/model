import os
import torch
import torch.nn.functional as F
from agent_model import DTSGModel
from sandbox_executor import SandboxExecutor
from transformers import AutoTokenizer

def train_grpo_sandbox(steps: int = 10, group_size: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[RL Train] Starting GRPO Sandbox training on: {device}")
    
    teacher_name = "Qwen/Qwen2.5-0.5B"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_SECRET")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, token=hf_token)
    
    model = DTSGModel(teacher_model_name=teacher_name, num_enzymes=6, max_loops=8).to(device)
    model.train()
    
    # Select parameters to optimize (only routing and planning heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    sandbox = SandboxExecutor(timeout=2.0)
    
    prompts = [
        "Write a Python script that calculates the sum of all prime numbers under 100 and prints it.",
        "Write a Python function to check if a string is a palindrome.",
        "Write a Python script to compute the Fibonacci sequence up to the 10th term."
    ]
    
    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        print(f"\n[Step {step+1}] Prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        # Generate a group of candidate paths/responses
        candidates = []
        log_probs_list = []
        rewards = []
        
        for g in range(group_size):
            # Sample candidate tokens without gradient tracking to save VRAM
            with torch.no_grad():
                generated = input_ids.clone()
                s = None
                teacher_cache = None
                for _ in range(32): # generate up to 32 tokens
                    logits, s, _, _, _, _, _, teacher_cache = model(generated, past_s=s, teacher_past_key_values=teacher_cache)
                    next_token_logits = logits[:, -1, :].clone()
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Sample token
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=-1)
            
            # Re-evaluate with gradients to get the trajectory log probs
            logits, s, policy_log_probs, _, _, _, _, _ = model(generated)
            seq_log_prob = torch.stack(policy_log_probs).mean() if policy_log_probs else torch.tensor(0.0, device=device)
            
            decoded_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            candidates.append(decoded_text)
            log_probs_list.append(seq_log_prob)
            
            # Evaluate using sandbox compiler feedback
            reward = 0.0
            if "def " in decoded_text or "print(" in decoded_text or "=" in decoded_text:
                # Extract code block and execute
                res = sandbox.run_python_code(decoded_text)
                if res["success"]:
                    reward += 1.5
                else:
                    reward -= 1.0 # Penalty for syntax / execution crash
            else:
                reward -= 0.5 # Penalty for not generating code
                
            rewards.append(reward)
            
        # Compute group-relative standardized rewards
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        relative_rewards = (rewards_tensor - mean_reward) / std_reward
        
        # Calculate policy gradient loss
        loss = 0.0
        for i in range(group_size):
            # policy gradient: -log_prob * relative_reward
            loss += -log_probs_list[i] * relative_rewards[i]
            
        loss = loss / group_size
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Group Rewards: {rewards} | Mean: {mean_reward.item():.4f} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_grpo_sandbox(steps=3, group_size=4)
