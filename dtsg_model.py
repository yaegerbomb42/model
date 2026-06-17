import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from transformers import AutoModelForCausalLM

class SwappableVirtualExperts:
    def __init__(self, filename="virtual_experts.bin", num_enzymes=6, num_experts=1024, num_heads=2, d_model=896, lora_rank=16):
        self.filename = filename
        self.num_enzymes = num_enzymes
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.d_model = d_model
        self.lora_rank = lora_rank
        
        self.gate_size = num_heads * d_model * lora_rank
        self.val_size = num_heads * d_model * lora_rank
        self.proj_size = num_heads * lora_rank * d_model
        self.expert_dim = self.gate_size + self.val_size + self.proj_size
        
        shape = (num_enzymes, num_experts, self.expert_dim)
        
        if not os.path.exists(filename):
            print(f"Pre-allocating 1024 virtual expert weights on disk: {filename}...")
            mmap_arr = np.memmap(filename, dtype='float32', mode='w+', shape=shape)
            mmap_arr[:] = np.random.normal(0, 0.02, size=shape).astype('float32')
            mmap_arr.flush()
            del mmap_arr
            
        self.mmap_arr = np.memmap(filename, dtype='float32', mode='r', shape=shape)
        # Keep a CPU copy
        self.cpu_weights = torch.from_numpy(np.array(self.mmap_arr))
        self.gpu_weights = None

    def get_expert_weights_gpu(self, enzyme_idx: int, indices_gpu: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.gpu_weights is None:
            print(f"[VRAM Experts] Transferring {self.filename} to {indices_gpu.device} ({dtype})...")
            self.gpu_weights = self.cpu_weights.to(device=indices_gpu.device, dtype=dtype)
        return self.gpu_weights[enzyme_idx, indices_gpu]

    def flush_to_disk(self):
        pass


class TopologyAuditor(nn.Module):
    """
    Monitors activation spikes for each enzyme node.
    If an enzyme's activations spike (exceed a threshold), we smoothly decay its routing gate mask
    towards 0 and ramp up a parallel linear random-projection bypass bridge.
    """
    def __init__(self, d_model: int, num_enzymes: int, threshold: float = 5.0, decay_rate: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.num_enzymes = num_enzymes
        self.threshold = threshold
        self.decay_rate = decay_rate
        
        self.register_buffer("activation_ema", torch.zeros(num_enzymes))
        self.momentum = 0.9
        
        self.bypass_bridges = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_enzymes)
        ])
        
        for bridge in self.bypass_bridges:
            nn.init.orthogonal_(bridge.weight, gain=0.1)
            nn.init.zeros_(bridge.bias)

    def update_and_mask(self, enzyme_idx: int, raw_output: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            current_norm = raw_output.norm(p=2, dim=-1).mean().item()
            if not math.isnan(current_norm) and not math.isinf(current_norm):
                self.activation_ema[enzyme_idx] = (
                    self.momentum * self.activation_ema[enzyme_idx] + (1 - self.momentum) * current_norm
                )
        
        tracked_norm = self.activation_ema[enzyme_idx]
        mask = torch.sigmoid(self.decay_rate * (self.threshold - tracked_norm)).to(dtype=raw_output.dtype)
        bypass_out = self.bypass_bridges[enzyme_idx](x_in)
        return mask * raw_output + (1.0 - mask) * bypass_out


class ContinuousMoEEnzyme(nn.Module):
    """
    Continuous Parameter Field Enzyme using a Direct Weight-Merging Router.
    Retrieves the mathematically perfect SVD weights of active Top-K experts from VRAM
    and computes a direct linear combination on-the-fly, requiring no training.
    """
    def __init__(self, d_model: int, enzyme_idx: int, virtual_experts: SwappableVirtualExperts, latent_dim: int = 7, num_experts: int = 1024, lora_rank: int = 16, num_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.enzyme_idx = enzyme_idx
        self.virtual_experts = virtual_experts
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.lora_rank = lora_rank
        self.num_heads = num_heads
        
        self.gate_size = num_heads * d_model * lora_rank
        self.val_size = num_heads * d_model * lora_rank
        self.proj_size = num_heads * lora_rank * d_model
        self.expert_dim = self.gate_size + self.val_size + self.proj_size
        
        # HyperNetwork to output continuous latent coordinates
        self.hyper_net = nn.Sequential(
            nn.Linear(d_model + d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, latent_dim)
        )
        
        # Trainable coordinate embeddings for the 1024 experts
        self.anchor_coords = nn.Parameter(torch.randn(num_experts, latent_dim) * 0.1)
        
        # GRU Gate cells for memory updates
        self.gru_gate_z = nn.Linear(d_model + d_model, d_model)
        self.gru_gate_r = nn.Linear(d_model + d_model, d_model)
        self.gru_gate_h = nn.Linear(d_model + d_model, d_model)

    def forward(self, x: torch.Tensor, s: torch.Tensor, ach_level: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        x_mean = x.mean(dim=1)
        s_mean = s.mean(dim=1)
        combined_features = torch.cat([x_mean, s_mean], dim=-1)
        z_target = self.hyper_net(combined_features)  # (batch_size, latent_dim)
        
        # Calculate distance to anchor coordinates to find the Top-K experts
        diff = z_target.unsqueeze(1) - self.anchor_coords.unsqueeze(0)  # (batch, 1024, latent_dim)
        distances = torch.norm(diff, p=2, dim=-1)  # (batch, 1024)
        
        # Determine dynamic K from ach_level
        if ach_level is not None:
            if isinstance(ach_level, torch.Tensor):
                ach_val = ach_level.item()
            else:
                ach_val = float(ach_level)
            
            if ach_val <= 1.5:
                K = 1
            elif ach_val <= 3.0:
                K = 2
            else:
                K = 3
        else:
            K = 2
            
        topk_distances, topk_indices = torch.topk(-distances, k=K, dim=-1)
        topk_probs = F.softmax(topk_distances, dim=-1)  # (batch, K)
        
        # Retrieve expert weights directly from GPU VRAM tensor
        active_weights = self.virtual_experts.get_expert_weights_gpu(self.enzyme_idx, topk_indices, x.dtype)  # (batch, K, expert_dim)
        
        # Linearly blend weights based on routing probabilities
        weights_tensor = torch.sum(topk_probs.unsqueeze(-1) * active_weights, dim=1)  # (batch, expert_dim)
        
        # Unpack blended weights
        gate_w = weights_tensor[:, :self.gate_size].view(batch_size, self.num_heads, self.d_model, self.lora_rank)
        val_w = weights_tensor[:, self.gate_size:self.gate_size+self.val_size].view(batch_size, self.num_heads, self.d_model, self.lora_rank)
        proj_w = weights_tensor[:, self.gate_size+self.val_size:].view(batch_size, self.num_heads, self.lora_rank, self.d_model)
        
        # SwiGLU FFN projection
        gate_out = torch.einsum('bsd,bhdr->bhsr', x, gate_w)
        val_out = torch.einsum('bsd,bhdr->bhsr', x, val_w)
        
        swish_val = val_out * torch.sigmoid(val_out)
        gated = swish_val * gate_out
        
        blend_x_out = torch.einsum('bhsr,bhrd->bhsd', gated, proj_w).sum(dim=1)
        
        x_out = x + blend_x_out
        
        # GRU Cell Update Step
        combined = torch.cat([x_out, s], dim=-1)
        z = torch.sigmoid(self.gru_gate_z(combined))
        r = torch.sigmoid(self.gru_gate_r(combined))
        candidate_combined = torch.cat([x_out, r * s], dim=-1)
        h_candidate = torch.tanh(self.gru_gate_h(candidate_combined))
        
        s_out = (1.0 - z) * s + z * h_candidate
        
        return x_out, s_out

    def get_orthonormal_loss(self) -> torch.Tensor:
        # Ensures anchors lie on orthogonal basis dimensions.
        norm_anchors = F.normalize(self.anchor_coords, p=2, dim=-1)
        similarity = torch.matmul(norm_anchors, norm_anchors.T)
        mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=self.anchor_coords.device)
        ortho_loss = torch.sum((similarity * mask.float()) ** 2)
        return ortho_loss / 1024.0


class DifferentiableRouter(nn.Module):
    def __init__(self, d_model: int, num_enzymes: int):
        super().__init__()
        self.gating = nn.Linear(d_model, num_enzymes)

    def forward(self, x: torch.Tensor, temp_scale: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mean = x.mean(dim=1)
        logits = self.gating(x_mean)
        if temp_scale is not None:
            logits = logits * temp_scale
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs, logits


class HybridKVMemory(nn.Module):
    """
    Hybrid Key-Value query layer that performs sparse retrieval over the history of loop iterations.
    """
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, history: list[torch.Tensor]) -> torch.Tensor:
        if not history:
            return x
            
        batch_size, seq_len, _ = x.shape
        num_loops = len(history)
        
        history_tensor = torch.cat(history, dim=1)
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(history_tensor).view(batch_size, num_loops * seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(history_tensor).view(batch_size, num_loops * seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(context)
        return x + output


class DTSGModel(nn.Module):
    def __init__(self, teacher_model_name: str = "gpt2", num_enzymes: int = 6, max_loops: int = 8, virtual_experts_path: str = "virtual_experts.bin", teacher_model: nn.Module = None):
        super().__init__()
        self.num_enzymes = num_enzymes
        self.max_loops = max_loops
        
        # 1. Load parameter seeds
        if teacher_model is not None:
            seed_model = teacher_model
        else:
            from transformers import AutoConfig
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_SECRET")
            config = AutoConfig.from_pretrained(teacher_model_name, token=hf_token)
            config.num_hidden_layers = 4
            # Keep seed_model on CPU if target is MPS. MPS fails to run F.embedding on parameters 
            # if the parent container object (the teacher model) has not been explicitly moved.
            # Storing it on CPU avoids this.
            device_target = "cpu" if torch.cuda.is_available() is False and torch.backends.mps.is_available() else "cpu"
            seed_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name, 
                config=config, 
                ignore_mismatched_sizes=True,
                token=hf_token
            )
            if device_target == "cpu":
                seed_model = seed_model.cpu()
        self.vocab_size = seed_model.config.vocab_size
        self.d_model = seed_model.config.hidden_size
        
        if hasattr(seed_model, "transformer") and hasattr(seed_model.transformer, "wte"):
            wte = seed_model.transformer.wte
        elif hasattr(seed_model, "model") and hasattr(seed_model.model, "embed_tokens"):
            wte = seed_model.model.embed_tokens
        else:
            raise AttributeError("Could not identify token embedding layer in the seed model.")
            
        lh = seed_model.lm_head
        
        # Keep the seed model frozen to extract context-rich intermediate representations
        # Wrap in a list so it is NOT registered as an nn.Module submodule,
        # preventing PyTorch from recursively copying it to GPU and inflating VRAM memory usage.
        self._teacher_model = [seed_model]
        self._embedding = [wte]
        self._lm_head = [lh]
        for param in seed_model.parameters():
            param.requires_grad = False
            
        self.position_embedding = nn.Embedding(2048, self.d_model)
        
        # Trainable projection from DTSG states back to the lm_head input space
        self.logits_proj = nn.Linear(self.d_model, self.d_model)
        nn.init.eye_(self.logits_proj.weight)
        nn.init.zeros_(self.logits_proj.bias)
        
        # 2. Graph routing components
        self.virtual_experts = SwappableVirtualExperts(filename=virtual_experts_path, num_enzymes=num_enzymes, d_model=self.d_model, num_heads=2, lora_rank=16)
        self.router = DifferentiableRouter(self.d_model, num_enzymes)
        self.topology_auditor = TopologyAuditor(self.d_model, num_enzymes)
        self.enzymes = nn.ModuleList([
            ContinuousMoEEnzyme(d_model=self.d_model, enzyme_idx=idx, virtual_experts=self.virtual_experts, latent_dim=7, lora_rank=16, num_heads=2)
            for idx in range(num_enzymes)
        ])
        
        # 3. Dynamic Entropy Gating Predictor
        self.entropy_gating = nn.Linear(self.d_model, 1)
        nn.init.zeros_(self.entropy_gating.weight)
        nn.init.zeros_(self.entropy_gating.bias)
        
        # 4. Gated Residual loops scaling factors
        self.residual_scale_x = nn.Parameter(torch.ones(1) * 0.1)
        self.residual_scale_s = nn.Parameter(torch.ones(1) * 0.1)
        
        # 5. Differentiable Value-Planning Head (Inline DMCTS)
        self.value_head = nn.Linear(self.d_model, 1)
        
        # 5b. Adaptive Halting Head (Dynamic Test-Time Compute)
        self.halting_head = nn.Linear(self.d_model, 1)
        nn.init.normal_(self.halting_head.weight, std=0.01)
        nn.init.constant_(self.halting_head.bias, -2.0)
        
        # 6. Contrastive projection head mapping latent s to 128D
        self.contrastive_proj = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # 7. Hybrid KV query layer
        self.kv_memory = HybridKVMemory(self.d_model)
        self.s_init = nn.Linear(self.d_model, self.d_model)
        
        # 8. Global Workspace Theory (GWT conscious bottleneck)
        self.workspace_projection = nn.Linear(self.d_model, self.d_model // 8)
        self.workspace_broadcast = nn.Linear(self.d_model // 8, self.d_model)
        nn.init.orthogonal_(self.workspace_projection.weight, gain=0.1)
        nn.init.orthogonal_(self.workspace_broadcast.weight, gain=0.1)

    def get_ponder_loss(self, halt_probs: torch.Tensor, p_target: float = 0.2) -> torch.Tensor:
        """
        Computes Ponder Net KL divergence regularization loss to prior Geometric(p_target).
        """
        # halt_probs shape: (num_loops, batch_size)
        num_loops, batch_size = halt_probs.shape
        device = halt_probs.device
        steps = torch.arange(num_loops, dtype=torch.float, device=device)
        prior = p_target * ((1.0 - p_target) ** steps)
        prior = prior / prior.sum().clamp(min=1e-10)
        prior = prior.unsqueeze(1).repeat(1, batch_size)
        clamped_probs = torch.clamp(halt_probs, min=1e-10)
        kl = halt_probs * (torch.log(clamped_probs) - torch.log(prior))
        return kl.sum(dim=0).mean()

    def forward(self, tokens: torch.Tensor, past_s: torch.Tensor = None, teacher_past_key_values = None) -> tuple:
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Extract pretrained contextual representations from layer 4 of the teacher model
        with torch.no_grad():
            teacher_device = next(self._teacher_model[0].parameters()).device
            if teacher_device != device:
                self._teacher_model[0] = self._teacher_model[0].to(device)
                teacher_device = device
                
            if teacher_past_key_values is not None:
                tokens_device = tokens[:, -1:].to(teacher_device)
            else:
                tokens_device = tokens.to(teacher_device)
                
            teacher_outs = self._teacher_model[0](
                tokens_device, 
                past_key_values=teacher_past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            x = teacher_outs.hidden_states[4].clone().to(device=device, dtype=self.logits_proj.weight.dtype)
            new_teacher_cache = teacher_outs.past_key_values
        
        if past_s is not None and past_s.shape == x.shape:
            s = past_s
        else:
            s = torch.tanh(self.s_init(x))
            
        kinetic_energy = torch.ones(batch_size, device=device, dtype=x.dtype)
        
        policy_log_probs = []
        x_history = []
        
        x_mean_init = x.mean(dim=1)
        temp_scale = 1.0 + F.softplus(self.entropy_gating(x_mean_init)) * 10.0
        
        p_remain = torch.ones(batch_size, device=device, dtype=x.dtype)
        halt_probs = []
        x_states = []
        s_states = []
        
        ach_history = []
        for loop in range(self.max_loops):
            # Inline planning & Neuromodulator estimation (Acetylcholine)
            with torch.no_grad():
                val_pred = self.value_head(x)  # (batch, seq_len, 1)
                current_val = val_pred.mean(dim=1)  # (batch, 1)
                ach_var = val_pred.var(dim=1).mean()
                if torch.isnan(ach_var):
                    ach_var = torch.tensor(0.0, device=device, dtype=val_pred.dtype)
                ach_level = 1.0 + torch.clamp(ach_var, min=0.0, max=5.0)
            
            ach_history.append(ach_level.item())
            probs, log_probs, logits = self.router(x, temp_scale=temp_scale * ach_level)
            
            # Inject value estimate back into gating logits to steer choice
            logits = logits + 0.1 * current_val
            probs = F.softmax(logits, dim=-1)
            
            clamped_probs = torch.clamp(probs, min=1e-10)
            entropy = -torch.sum(probs * torch.log(clamped_probs), dim=-1)
            
            energy_loss = 0.2 + 0.1 * entropy
            kinetic_energy = kinetic_energy - energy_loss
            
            active_mask = (kinetic_energy > 0.0).to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
            policy_log_probs.append(log_probs)
            
            x_history.append(x.clone())
            
            node_x_outputs = []
            node_s_outputs = []
            
            for idx, enzyme in enumerate(self.enzymes):
                ex, es = enzyme(x, s, ach_level=ach_level)
                ex_blended = self.topology_auditor.update_and_mask(idx, ex, x)
                node_x_outputs.append(ex_blended)
                node_s_outputs.append(es)
                
            stacked_x = torch.stack(node_x_outputs, dim=0)
            stacked_s = torch.stack(node_s_outputs, dim=0)
            
            probs_unsqueezed = probs.T.unsqueeze(-1).unsqueeze(-1)
            x_step = torch.sum(stacked_x * probs_unsqueezed, dim=0)
            s_step = torch.sum(stacked_s * probs_unsqueezed, dim=0)
            
            x = x + self.residual_scale_x * (active_mask * x_step)
            s = s + self.residual_scale_s * (active_mask * s_step)
            
            # Global Workspace Broadcast (Bottleneck attention bottleneck)
            x_pool = x.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
            workspace = self.workspace_projection(x_pool)  # (batch, 1, d_model // 8)
            broadcast_sig = torch.tanh(self.workspace_broadcast(workspace))  # (batch, 1, d_model)
            x = x + 0.1 * broadcast_sig  # Broadcast back to token pathways
            
            # Adaptive halting calculation
            if loop == self.max_loops - 1:
                p_halt = p_remain
            else:
                h_l = self.halting_head(x.mean(dim=1)).squeeze(-1)
                p_l = torch.sigmoid(h_l)
                p_halt = p_remain * p_l
                p_remain = p_remain * (1.0 - p_l)
                
            halt_probs.append(p_halt)
            x_states.append(x.clone())
            s_states.append(s.clone())
            
            if not self.training and (p_remain < 0.05).all():
                break
                
        # Stack and blend states based on halting probability distribution
        stacked_probs = torch.stack(halt_probs, dim=0)  # (num_loops, batch)
        stacked_probs = stacked_probs / stacked_probs.sum(dim=0, keepdim=True).clamp(min=1e-10)
        
        probs_expanded = stacked_probs.unsqueeze(-1).unsqueeze(-1)  # (num_loops, batch, 1, 1)
        x_final = torch.sum(torch.stack(x_states, dim=0) * probs_expanded, dim=0)
        s_final = torch.sum(torch.stack(s_states, dim=0) * probs_expanded, dim=0)
        
        x_final = self.kv_memory(x_final, x_history)
        
        # Project back to late-stage representation space before lm_head
        x_final = self.logits_proj(x_final)
        
        s_contrastive = self.contrastive_proj(s_final.mean(dim=1))
        # Move lm_head to the input's device and dtype to avoid device/dtype mismatch
        head_device = next(self._lm_head[0].parameters()).device
        if head_device != x_final.device or next(self._lm_head[0].parameters()).dtype != x_final.dtype:
            self._lm_head[0] = self._lm_head[0].to(device=x_final.device, dtype=x_final.dtype)
        logits_vocab = self._lm_head[0](x_final)
        
        # Compute ponder loss
        ponder_loss = self.get_ponder_loss(stacked_probs)
        mean_ach = sum(ach_history) / len(ach_history)
        
        return logits_vocab, s_final, policy_log_probs, s_contrastive, ponder_loss, halt_probs, mean_ach, new_teacher_cache

    def get_orthonormal_penalty(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.logits_proj.weight.device)

    def sync_virtual_experts_gradients(self, lr: float = 1e-3):
        pass

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        generated = prompt_tokens.clone()
        s = None
        teacher_cache = None
        for _ in range(max_new_tokens):
            logits, s, _, _, _, _, _, teacher_cache = self(generated, past_s=s, teacher_past_key_values=teacher_cache)
            next_token_logits = logits[:, -1, :].clone() / temperature
            for token_id in set(generated[0].tolist()):
                logit = next_token_logits[0, token_id]
                if logit > 0:
                    next_token_logits[0, token_id] /= 1.2
                else:
                    next_token_logits[0, token_id] *= 1.2
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated

    @torch.no_grad()
    def generate_mcts(self, prompt_tokens: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, lookahead_depth: int = 3, branching_factor: int = 3) -> torch.Tensor:
        """
        Executes Monte Carlo Tree Search at lookahead steps to find the most coherent token generation path.
        """
        generated = prompt_tokens.clone()
        s = None
        teacher_cache = None
        
        import copy
        def clone_cache(cache):
            if cache is None:
                return None
            try:
                from transformers.cache_utils import DynamicCache
                if isinstance(cache, DynamicCache):
                    new_cache = DynamicCache()
                    new_cache.key_cache = [t.clone() for t in cache.key_cache]
                    new_cache.value_cache = [t.clone() for t in cache.value_cache]
                    new_cache._seen_tokens = cache._seen_tokens
                    return new_cache
            except Exception:
                pass
            try:
                return copy.deepcopy(cache)
            except Exception:
                return tuple(
                    tuple(t.clone() if t is not None else None for t in layer)
                    if layer is not None else None
                    for layer in cache
                )
            
        for _ in range(max_new_tokens):
            logits, s, _, _, _, _, _, teacher_cache = self(generated, past_s=s, teacher_past_key_values=teacher_cache)
            next_token_logits = logits[:, -1, :].clone() / temperature
            for token_id in set(generated[0].tolist()):
                logit = next_token_logits[0, token_id]
                if logit > 0:
                    next_token_logits[0, token_id] /= 1.2
                else:
                    next_token_logits[0, token_id] *= 1.2
            probs = F.softmax(next_token_logits, dim=-1)
            
            top_probs, top_candidates = torch.topk(probs, k=branching_factor, dim=-1)
            
            best_candidate = None
            best_value = -float('inf')
            
            for i in range(branching_factor):
                candidate_token = top_candidates[:, i:i+1]
                
                rollout_tokens = torch.cat([generated, candidate_token], dim=-1)
                rollout_s = s.clone() if s is not None else None
                rollout_cache = clone_cache(teacher_cache)
                
                rollout_val_sum = 0.0
                
                for d in range(lookahead_depth):
                    r_logits, rollout_s, _, _, _, _, _, rollout_cache = self(rollout_tokens, past_s=rollout_s, teacher_past_key_values=rollout_cache)
                    val_estimate = self.value_head(rollout_s).mean().item()
                    rollout_val_sum += val_estimate
                    
                    next_r_token = torch.argmax(r_logits[:, -1, :], dim=-1, keepdim=True)
                    rollout_tokens = torch.cat([rollout_tokens, next_r_token], dim=-1)
                
                candidate_score = rollout_val_sum + torch.log(top_probs[:, i]).item()
                
                if candidate_score > best_value:
                    best_value = candidate_score
                    best_candidate = candidate_token
            
            generated = torch.cat([generated, best_candidate], dim=-1)
            
        return generated
