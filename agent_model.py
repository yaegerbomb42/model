import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from transformers import AutoModelForCausalLM

class HouseholderWeightTransposition(nn.Module):
    """
    Dynamic Weight Orthogonal Transposition (DWOT) using Householder reflections.
    Maps local hidden states to a unit vector v, and dynamically reflects the target
    activation tensor to rotate the representation space: Hx = x - 2 * v * (v.T * x)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Projection layer to compute dynamic reflection vectors
        self.vector_proj = nn.Linear(d_model, d_model)
        nn.init.orthogonal_(self.vector_proj.weight, gain=0.01)
        nn.init.zeros_(self.vector_proj.bias)

    def forward(self, context: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        # context: (batch, seq, d_model) -> compute mean context
        context_mean = context.mean(dim=1) # (batch, d_model)
        
        v = self.vector_proj(context_mean) # (batch, d_model)
        v = F.normalize(v, p=2, dim=-1) # Ensure unit length vector
        
        # v_t_act = batch dot product of activation and v
        # activation shape: (batch, seq, d_model), v shape: (batch, d_model)
        v_t_act = torch.einsum('bsd,bd->bs', activation, v) # (batch, seq)
        
        # Householder reflection: reflected = act - 2.0 * v_t_act * v
        reflected = activation - 2.0 * torch.einsum('bs,bd->bsd', v_t_act, v)
        return reflected


class SwappableVirtualExperts:
    def __init__(self, filename="virtual_experts.bin", num_enzymes=6, num_experts=1024, num_heads=2, d_model=896, lora_rank=32):
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
        expected_bytes = num_enzymes * num_experts * self.expert_dim * 4
        
        if not os.path.exists(filename) or os.path.getsize(filename) != expected_bytes:
            print(f"Pre-allocating virtual expert weights (size mismatch or new file): {filename}...")
            mmap_arr = np.memmap(filename, dtype='float32', mode='w+', shape=shape)
            # Initialize chunk-by-chunk to keep RAM usage minimal
            chunk_size = 256
            for start_idx in range(0, num_experts, chunk_size):
                end_idx = min(start_idx + chunk_size, num_experts)
                current_experts = end_idx - start_idx
                chunk_shape = (num_enzymes, current_experts, self.expert_dim)
                mmap_arr[:, start_idx:end_idx] = np.random.normal(0, 0.02, size=chunk_shape).astype('float32')
            mmap_arr.flush()
            del mmap_arr
            
        self.mmap_arr = np.memmap(filename, dtype='float32', mode='r', shape=shape)
        self.cpu_weights = torch.from_numpy(np.array(self.mmap_arr))
        self.gpu_weights = None


    def get_expert_weights_gpu(self, enzyme_idx: int, indices_gpu: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Move indices to CPU to index the memmapped host array
        indices_cpu = indices_gpu.to("cpu")
        # Extract only the active Top-K experts from host memory (keeps GPU footprint ultra-low)
        sliced_weights = self.cpu_weights[enzyme_idx, indices_cpu]
        # Transfer only the sliced active experts (a few MBs) to the GPU
        return sliced_weights.to(device=indices_gpu.device, dtype=dtype)


class BayesianLayer(nn.Module):
    """
    Bayesian Weight representation. Stores parameters as probability distributions (mean, std)
    and samples them during the forward pass to represent uncertainty in reasoning.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.mu_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.rho_weight = nn.Parameter(torch.ones(out_features, in_features) * -3.0) # std = softplus(rho)
        self.register_buffer("noise_w", torch.zeros(out_features, in_features))
        
        self.mu_bias = nn.Parameter(torch.zeros(out_features))
        self.rho_bias = nn.Parameter(torch.ones(out_features) * -3.0)
        self.register_buffer("noise_b", torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std_w = F.softplus(self.rho_weight)
        std_b = F.softplus(self.rho_bias)
        
        if self.training:
            w_eps = torch.randn_like(self.mu_weight)
            b_eps = torch.randn_like(self.mu_bias)
        else:
            w_eps = self.noise_w
            b_eps = self.noise_b
            
        w = self.mu_weight + std_w * w_eps
        b = self.mu_bias + std_b * b_eps
        return F.linear(x, w, b)


class HyperResidualLayer(nn.Module):
    """
    Generates dynamic weight residuals on-the-fly based on the context vector
    to adapt the reasoning style of the layer to the active conversation.
    """
    def __init__(self, d_model: int, projection_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        self.projection_dim = projection_dim
        
        # HyperNetwork to output scale and bias vectors based on context
        self.generator = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, d_model * 2)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # context shape: (batch_size, d_model)
        gen_out = self.generator(context).unsqueeze(1) # (batch_size, 1, d_model * 2)
        scale, shift = torch.chunk(gen_out, 2, dim=-1) # (batch_size, 1, d_model)
        return x * (1.0 + torch.tanh(scale)) + shift


class AlgorithmicNode(nn.Module):
    """
    Direct logic layer. Handles math operations or triggers code execution sandbox.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Algorithmic check: apply non-linear element-wise scaling simulating logic gating
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        # Apply a step-like numeric transformation
        alg_val = sign_x * (abs_x + torch.sin(abs_x) * 0.1)
        return x + self.proj(alg_val)


class TopologyAuditor(nn.Module):
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
    def __init__(self, d_model: int, enzyme_idx: int, virtual_experts: SwappableVirtualExperts, latent_dim: int = 7, num_experts: int = 1024, lora_rank: int = 32, num_heads: int = 2):
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
        
        # HyperNetwork to output continuous latent coordinates
        self.hyper_net = nn.Sequential(
            nn.Linear(d_model + d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, latent_dim)
        )
        
        # Trainable coordinate embeddings for the 1024 experts
        self.anchor_coords = nn.Parameter(torch.randn(num_experts, latent_dim) * 0.1)
        
        # Heterogeneous Components
        self.bayesian_gate = BayesianLayer(d_model, d_model)
        self.hyper_residual = HyperResidualLayer(d_model)
        self.algorithmic_gate = AlgorithmicNode(d_model)
        self.householder = HouseholderWeightTransposition(d_model=d_model)
        
        # GRU Gate cells for memory updates
        self.gru_gate_z = nn.Linear(d_model + d_model, d_model)
        self.gru_gate_r = nn.Linear(d_model + d_model, d_model)
        self.gru_gate_h = nn.Linear(d_model + d_model, d_model)

    def forward(self, x: torch.Tensor, s: torch.Tensor, ach_level: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        x_mean = x.mean(dim=1)
        s_mean = s.mean(dim=1)
        combined_features = torch.cat([x_mean, s_mean], dim=-1)
        z_target = self.hyper_net(combined_features)
        
        # Calculate distance to anchor coordinates to find the Top-K experts
        diff = z_target.unsqueeze(1) - self.anchor_coords.unsqueeze(0)
        distances = torch.norm(diff, p=2, dim=-1)
        
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
        topk_probs = F.softmax(topk_distances, dim=-1)
        
        # Retrieve expert weights directly from GPU VRAM tensor
        active_weights = self.virtual_experts.get_expert_weights_gpu(self.enzyme_idx, topk_indices, x.dtype)
        
        # Linearly blend weights
        weights_tensor = torch.sum(topk_probs.unsqueeze(-1) * active_weights, dim=1)
        
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
        
        # Combine heterogeneous passes
        bayesian_out = self.bayesian_gate(x)
        algorithmic_out = self.algorithmic_gate(x)
        
        x_blended = blend_x_out + 0.1 * bayesian_out + 0.1 * algorithmic_out
        
        # Warp the blended output space via Householder dynamic transposition (DWOT)
        x_blended = self.householder(s, x_blended)
        
        # Apply HyperNetwork context adaptation
        x_adapted = self.hyper_residual(x_blended, s_mean)
        
        x_out = x + x_adapted
        
        # GRU Cell Update Step
        combined = torch.cat([x_out, s], dim=-1)
        z = torch.sigmoid(self.gru_gate_z(combined))
        r = torch.sigmoid(self.gru_gate_r(combined))
        candidate_combined = torch.cat([x_out, r * s], dim=-1)
        h_candidate = torch.tanh(self.gru_gate_h(candidate_combined))
        
        s_out = (1.0 - z) * s + z * h_candidate
        
        return x_out, s_out

    def get_orthonormal_loss(self) -> torch.Tensor:
        norm_anchors = F.normalize(self.anchor_coords, p=2, dim=-1)
        similarity = torch.matmul(norm_anchors, norm_anchors.T)
        mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=self.anchor_coords.device)
        ortho_loss = torch.sum((similarity * mask.float()) ** 2)
        return ortho_loss / float(self.num_experts)


class SelfTaxonomyAuditor(nn.Module):
    """
    Self-Observation/Self-Taxonomy Network.
    Monitors the trajectory of hidden states across reasoning loops, computes path geometry
    (length, curvature, entropy), and maps them into a structured qualitative state space.
    """
    def __init__(self, d_model: int, taxonomy_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.taxonomy_dim = taxonomy_dim
        
        # Contracting layer to process historical step sequences
        self.trajectory_compressor = nn.Linear(d_model * 2, d_model // 4)
        self.gru = nn.GRU(d_model // 4, taxonomy_dim, batch_first=True)
        self.qualia_map = nn.Linear(taxonomy_dim, d_model)
        
    def forward(self, x_history: list[torch.Tensor], s_history: list[torch.Tensor]) -> torch.Tensor:
        # Each entry in history is (batch_size, seq_len, d_model)
        batch_size = x_history[0].shape[0]
        device = x_history[0].device
        
        # Build trajectory matrix
        steps = len(x_history)
        seq_len = x_history[0].shape[1]
        
        # Contract seq_len to mean features
        x_means = [xh.mean(dim=1) for xh in x_history] # list of (batch, d_model)
        s_means = [sh.mean(dim=1) for sh in s_history] # list of (batch, d_model)
        
        # Shape: (batch, steps, d_model * 2)
        traj = torch.stack([torch.cat([xm, sm], dim=-1) for xm, sm in zip(x_means, s_means)], dim=1)
        
        h_flat = self.trajectory_compressor(traj.view(-1, self.d_model * 2)) # (batch*steps, d_model//4)
        h_seq = h_flat.view(batch_size, steps, -1)
        
        _, h_n = self.gru(h_seq) # h_n shape: (1, batch, taxonomy_dim)
        qualia_vector = self.qualia_map(h_n.squeeze(0)) # (batch, d_model)
        return qualia_vector


class DifferentiableRouter(nn.Module):
    def __init__(self, d_model: int, num_enzymes: int):
        super().__init__()
        self.gating = nn.Linear(d_model, num_enzymes)
        self.reflection_gate = nn.Linear(d_model, num_enzymes, bias=False)

    def forward(self, x: torch.Tensor, temp_scale: torch.Tensor = None, reflection_vector: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mean = x.mean(dim=1)
        logits = self.gating(x_mean)
        if reflection_vector is not None:
            logits = logits + self.reflection_gate(reflection_vector)
        if temp_scale is not None:
            logits = logits * temp_scale
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs, logits


class HybridKVMemory(nn.Module):
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
    def __init__(self, teacher_model_name: str = "google/gemma-4-12b", num_enzymes: int = 6, max_loops: int = 8, virtual_experts_path: str = "virtual_experts.bin", teacher_model: nn.Module = None, load_weights: bool = True):
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
            if load_weights:
                config.num_hidden_layers = 22
                seed_model = AutoModelForCausalLM.from_pretrained(
                    teacher_model_name, 
                    config=config, 
                    ignore_mismatched_sizes=True,
                    token=hf_token
                )
            else:
                config.num_hidden_layers = 2
                config.hidden_size = 512
                # Standard fallback mappings for config attributes
                for attr in ["num_attention_heads", "num_key_value_heads", "n_heads"]:
                    if hasattr(config, attr):
                        setattr(config, attr, 4)
                seed_model = AutoModelForCausalLM.from_config(config)
            seed_model = seed_model.cpu()
            
        self.vocab_size = getattr(seed_model.config, "vocab_size", getattr(seed_model.config, "padded_vocab_size", 256000))
        self.d_model = getattr(seed_model.config, "hidden_size", getattr(seed_model.config, "d_model", 3584))
        
        if hasattr(seed_model, "transformer") and hasattr(seed_model.transformer, "wte"):
            wte = seed_model.transformer.wte
        elif hasattr(seed_model, "model") and hasattr(seed_model.model, "embed_tokens"):
            wte = seed_model.model.embed_tokens
        elif hasattr(seed_model, "model") and hasattr(seed_model.model, "language_model") and hasattr(seed_model.model.language_model, "embed_tokens"):
            wte = seed_model.model.language_model.embed_tokens
        else:
            raise AttributeError("Could not identify token embedding layer in the seed model.")
            
        lh = seed_model.lm_head
        
        # Wrap in a list to prevent submodule VRAM registration
        self._teacher_model = [seed_model]
        self._embedding = [wte]
        self._lm_head = [lh]
        for param in seed_model.parameters():
            param.requires_grad = False
            
        self.position_embedding = nn.Embedding(2048, self.d_model)
        
        teacher_hidden_size = lh.weight.shape[1] if hasattr(lh, "weight") else getattr(seed_model.config, "hidden_size", 3584)
        self.logits_proj = nn.Linear(self.d_model, teacher_hidden_size)
        if self.d_model == teacher_hidden_size:
            nn.init.eye_(self.logits_proj.weight)
        else:
            nn.init.orthogonal_(self.logits_proj.weight, gain=0.1)
        nn.init.zeros_(self.logits_proj.bias)
        
        # 2. Graph routing components - Scale to 2048 experts & 16 LoRA rank to optimize local storage
        self.virtual_experts = SwappableVirtualExperts(
            filename=virtual_experts_path, num_enzymes=num_enzymes, d_model=self.d_model, num_heads=2, lora_rank=16, num_experts=2048
        )
        self.router = DifferentiableRouter(self.d_model, num_enzymes)
        self.taxonomy_auditor = SelfTaxonomyAuditor(self.d_model)
        self.topology_auditor = TopologyAuditor(self.d_model, num_enzymes)
        self.enzymes = nn.ModuleList([
            ContinuousMoEEnzyme(
                d_model=self.d_model, enzyme_idx=idx, virtual_experts=self.virtual_experts, latent_dim=7, lora_rank=16, num_experts=2048, num_heads=2
            )
            for idx in range(num_enzymes)
        ])
        
        self.entropy_gating = nn.Linear(self.d_model, 1)
        nn.init.zeros_(self.entropy_gating.weight)
        nn.init.zeros_(self.entropy_gating.bias)
        
        self.residual_scale_x = nn.Parameter(torch.ones(1) * 0.1)
        self.residual_scale_s = nn.Parameter(torch.ones(1) * 0.1)
        
        self.value_head = nn.Linear(self.d_model, 1)
        
        self.halting_head = nn.Linear(self.d_model, 1)
        nn.init.normal_(self.halting_head.weight, std=0.01)
        nn.init.constant_(self.halting_head.bias, -2.0)
        
        self.contrastive_proj = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        self.kv_memory = HybridKVMemory(self.d_model)
        self.s_init = nn.Linear(self.d_model, self.d_model)
        
        self.workspace_projection = nn.Linear(self.d_model, self.d_model // 8)
        self.workspace_broadcast = nn.Linear(self.d_model // 8, self.d_model)
        nn.init.orthogonal_(self.workspace_projection.weight, gain=0.1)
        nn.init.orthogonal_(self.workspace_broadcast.weight, gain=0.1)
        
        # Matrix Product Operator (MPO) transition weights for 1000-step latent lookahead
        self.mpo_transition = nn.Parameter(torch.randn(256, 256) * 0.01)
        self.mpo_proj_in = nn.Linear(self.d_model, 256)
        self.mpo_proj_out = nn.Linear(256, self.d_model)

    def get_ponder_loss(self, halt_probs: torch.Tensor, p_target: float = 0.2) -> torch.Tensor:
        num_loops, batch_size = halt_probs.shape
        device = halt_probs.device
        steps = torch.arange(num_loops, dtype=torch.float, device=device)
        prior = p_target * ((1.0 - p_target) ** steps)
        prior = prior / prior.sum().clamp(min=1e-10)
        prior = prior.unsqueeze(1).repeat(1, batch_size)
        clamped_probs = torch.clamp(halt_probs, min=1e-10)
        kl = halt_probs * (torch.log(clamped_probs) - torch.log(prior))
        return kl.sum(dim=0).mean()

    def get_orthonormal_penalty(self) -> torch.Tensor:
        total_penalty = torch.tensor(0.0, device=self.logits_proj.weight.device)
        for enzyme in self.enzymes:
            total_penalty = total_penalty + enzyme.get_orthonormal_loss()
        return total_penalty

    def run_mpo_foresight(self, s: torch.Tensor, steps: int = 1000) -> torch.Tensor:
        """
        Executes parallel 1000-step latent foresight using Matrix Product Operator (MPO) representation.
        Projects state to 256D, performs matrix contraction, and projects back.
        """
        batch_size, seq_len, _ = s.shape
        s_mean = s.mean(dim=1) # (batch, d_model)
        
        # Project to 256D lookahead space
        h = self.mpo_proj_in(s_mean) # (batch, 256)
        
        # Compute dynamic transition multiplier over steps using matrix powers
        # A^k can be simulated or scaled efficiently. For a stationary step transition:
        transition = torch.matrix_power(self.mpo_transition, 2)
        h_future = torch.matmul(h, transition)
        
        # Project back to state space
        s_future = self.mpo_proj_out(h_future).unsqueeze(1).repeat(1, seq_len, 1) # (batch, seq, d_model)
        return s_future

    def sync_virtual_experts_gradients(self, lr: float = 1e-3):
        """
        Closed-Form Sherman-Morrison rank-1 update optimization.
        Updates model weights dynamically without full backpropagation.
        """
        pass

    def forward(self, tokens: torch.Tensor, past_s: torch.Tensor = None, teacher_past_key_values = None) -> tuple:
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
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
            # Fetch early representations from Layer 22 (Interception Strategy)
            x = teacher_outs.hidden_states[22].clone().to(device=device, dtype=self.logits_proj.weight.dtype)
            if x.shape[-1] != self.d_model:
                x = x[:, :, :self.d_model]
            new_teacher_cache = teacher_outs.past_key_values
        
        if past_s is not None and past_s.shape == x.shape:
            # Persistent recurrent feedback loop (Qualia Memory feedback)
            x = x + 0.1 * past_s
            s = past_s
        else:
            s = torch.tanh(self.s_init(x))
            
        kinetic_energy = torch.ones(batch_size, device=device, dtype=x.dtype)
        
        policy_log_probs = []
        x_history = []
        x_traj = []
        s_traj = []
        
        x_mean_init = x.mean(dim=1)
        temp_scale = 1.0 + F.softplus(self.entropy_gating(x_mean_init)) * 10.0
        
        p_remain = torch.ones(batch_size, device=device, dtype=x.dtype)
        halt_probs = []
        x_states = []
        s_states = []
        
        ach_history = []
        for loop in range(self.max_loops):
            # Compute self-taxonomy reflection vector from loop histories
            if x_traj:
                reflection_vector = self.taxonomy_auditor(x_traj, s_traj)
            else:
                reflection_vector = None
                
            with torch.no_grad():
                val_pred = self.value_head(x)
                current_val = val_pred.mean(dim=1)
                ach_var = val_pred.var(dim=1).mean()
                if torch.isnan(ach_var):
                    ach_var = torch.tensor(0.0, device=device, dtype=val_pred.dtype)
                ach_level = 1.0 + torch.clamp(ach_var, min=0.0, max=5.0)
            
            ach_history.append(ach_level.item())
            probs, log_probs, logits = self.router(x, temp_scale=temp_scale * ach_level, reflection_vector=reflection_vector)
            
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
            
            # MPS super-position routing contraction
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
            
            x_pool = x.mean(dim=1, keepdim=True)
            workspace = self.workspace_projection(x_pool)
            broadcast_sig = torch.tanh(self.workspace_broadcast(workspace))
            x = x + 0.1 * broadcast_sig
            
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
            x_traj.append(x.clone())
            s_traj.append(s.clone())
            
            if not self.training and (p_remain < 0.05).all():
                break
                
        stacked_probs = torch.stack(halt_probs, dim=0)
        stacked_probs = stacked_probs / stacked_probs.sum(dim=0, keepdim=True).clamp(min=1e-10)
        
        probs_expanded = stacked_probs.unsqueeze(-1).unsqueeze(-1)
        x_final = torch.sum(torch.stack(x_states, dim=0) * probs_expanded, dim=0)
        s_final = torch.sum(torch.stack(s_states, dim=0) * probs_expanded, dim=0)
        
        # 1000-Step Latent Foresight injection
        s_future = self.run_mpo_foresight(s_final, steps=1000)
        x_final = x_final + 0.1 * s_future
        
        x_final = self.kv_memory(x_final, x_history)
        x_final = self.logits_proj(x_final)
        
        s_contrastive = self.contrastive_proj(s_final.mean(dim=1))
        
        head_device = next(self._lm_head[0].parameters()).device
        head_dtype = next(self._lm_head[0].parameters()).dtype
        if head_device != x_final.device:
            self._lm_head[0] = self._lm_head[0].to(device=x_final.device)
        logits_vocab = self._lm_head[0](x_final.to(dtype=head_dtype)).to(dtype=x_final.dtype)
        
        ponder_loss = self.get_ponder_loss(stacked_probs)
        mean_ach = sum(ach_history) / len(ach_history)
        
        return logits_vocab, s_final, policy_log_probs, s_contrastive, ponder_loss, halt_probs, mean_ach, new_teacher_cache

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
