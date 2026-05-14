"""
Adaptive Sink Absorption (ASA) on top of LlamaWithRegisters.

Two complementary mechanisms:
  1. Position-0 registers (existing): absorb BOS-style positional attention sinks
  2. Per-layer AdaptiveSinkGate (new): absorb semantic attention sinks

AdaptiveSinkGate logic
----------------------
After each transformer layer we inspect the hidden-state L2 norm distribution.
Tokens whose norm z-score exceeds a learnable threshold are flagged as
"over-attended sinks". We soft-project out the component along learnable
absorber directions — up to max_alpha (default 0.5) of that component.

This means:
  - If a token is semantically important (high, but not anomalously high norm)
    → gate stays closed, full attention preserved.
  - If a token is an attention dump (norm >> mean + 2σ)
    → gate opens proportionally, excess is softly removed.

Threshold is per-layer learnable so the model discovers the right sensitivity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from register_llama.model import LlamaWithRegisters
from register_llama.config import Config


class AdaptiveSinkGate(nn.Module):
    """
    Inserted after each LlamaDecoderLayer output.

    Detects over-attended tokens via hidden-state norm z-score and
    soft-removes their excess component along learnable absorber directions.

    Parameters
    ----------
    hidden_size   : model hidden dimension
    num_absorbers : number of learnable absorber directions (default 4)
    z_threshold   : initial z-score cutoff (learnable, default 2.0)
    max_alpha     : maximum fraction of sink component to remove (default 0.5)
    """

    def __init__(
        self,
        hidden_size: int,
        num_absorbers: int = 4,
        z_threshold: float = 2.0,
        max_alpha: float = 0.5,
    ):
        super().__init__()
        self.num_absorbers = num_absorbers
        self.max_alpha = max_alpha

        # Learnable absorber directions in hidden space
        self.absorber_dirs = nn.Parameter(
            torch.randn(num_absorbers, hidden_size) * 0.02
        )
        # Log-parameterized threshold (always positive)
        self.log_z_threshold = nn.Parameter(torch.tensor(z_threshold).log())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states : (B, T, H)
        Returns corrected hidden_states of same shape.
        """
        # 1. Per-token L2 norm
        norms = hidden_states.norm(dim=-1)          # (B, T)

        # 2. Z-score across the sequence dimension
        mean_n = norms.mean(dim=-1, keepdim=True)   # (B, 1)
        std_n  = norms.std(dim=-1,  keepdim=True).clamp(min=1e-6)
        z = (norms - mean_n) / std_n                # (B, T)

        # 3. Soft gate: only activates above threshold, capped at max_alpha
        z_thr = self.log_z_threshold.exp()
        excess = (z - z_thr).clamp(min=0)           # (B, T) ≥ 0
        alpha  = torch.tanh(excess) * self.max_alpha # (B, T) ∈ [0, max_alpha]

        # 4. Normalised absorber directions
        dirs = F.normalize(self.absorber_dirs, dim=-1)  # (K, H)

        # 5. Project hidden state onto absorber subspace and remove alpha fraction
        #    proj = Σ_k (h · dir_k) dir_k
        proj_coeffs = hidden_states @ dirs.T             # (B, T, K)
        absorbed    = proj_coeffs @ dirs                 # (B, T, H)

        corrected = hidden_states - alpha.unsqueeze(-1) * absorbed

        return corrected


class LlamaDecoderLayerWithASG(nn.Module):
    """
    Thin wrapper: runs the original LlamaDecoderLayer then applies ASG.
    """

    def __init__(self, base_layer: nn.Module, asg: AdaptiveSinkGate):
        super().__init__()
        self.base_layer = base_layer
        self.asg = asg

    def forward(self, hidden_states, **kwargs):
        outputs = self.base_layer(hidden_states, **kwargs)
        # outputs[0] is the updated hidden states
        corrected = self.asg(outputs[0])
        return (corrected,) + outputs[1:]


class LlamaWithAdaptiveRegisters(LlamaWithRegisters):
    """
    LlamaWithRegisters + per-layer AdaptiveSinkGate.

    Architecture:
      Input → [BOS][REG1..4][tokens]   (all pos-0 for BOS+REGs)
          ↓  existing position-0 registers absorb BOS-style sink
      Each transformer layer output → AdaptiveSinkGate
          ↓  absorbs semantic sinks (norm z-score > threshold)
      LM head
    """

    def __init__(self, config):
        super().__init__(config)
        hidden = config.hidden_size
        num_abs  = getattr(config, "num_asg_absorbers",  4)
        z_thr    = getattr(config, "asg_z_threshold",    2.0)
        max_alph = getattr(config, "asg_max_alpha",      0.5)

        # Wrap every decoder layer with an ASG
        new_layers = nn.ModuleList()
        for layer in self.model.layers:
            asg = AdaptiveSinkGate(hidden, num_abs, z_thr, max_alph)
            new_layers.append(LlamaDecoderLayerWithASG(layer, asg))
        self.model.layers = new_layers

    # Forward is inherited from LlamaWithRegisters (position-0 injection + chunked CE)


# ── Setup helpers ────────────────────────────────────────────────────────────

def load_adaptive_model(cfg: Config, tokenizer, device_map="auto"):
    """Load LlamaWithAdaptiveRegisters and initialise embeddings."""
    from register_llama.model import _init_register_embeddings

    model_cfg = AutoConfig.from_pretrained(cfg.model_path)
    model_cfg.num_registers      = cfg.num_registers
    model_cfg.num_asg_absorbers  = cfg.num_asg_absorbers
    model_cfg.asg_z_threshold    = cfg.asg_z_threshold
    model_cfg.asg_max_alpha      = cfg.asg_max_alpha

    model = LlamaWithAdaptiveRegisters.from_pretrained(
        cfg.model_path,
        config=model_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    orig_vocab = model_cfg.vocab_size
    model.resize_token_embeddings(len(tokenizer))
    _init_register_embeddings(model, tokenizer, cfg, orig_vocab)

    return model, orig_vocab


def apply_lora_adaptive(model, cfg: Config, orig_vocab_size, tokenizer):
    """
    Apply LoRA to attention projections.
    ASG parameters (absorber_dirs, log_z_threshold) are trained directly
    (not via LoRA) since they are small and already random-initialised.
    Embedding is frozen as before.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # Freeze embedding (same reason as base model)
    model.get_input_embeddings().weight.requires_grad_(False)

    # Ensure ASG params are trainable (PEFT may freeze non-target params)
    asg_params = 0
    for name, param in model.named_parameters():
        if "asg" in name or "absorber" in name or "log_z" in name:
            param.requires_grad_(True)
            asg_params += param.numel()

    model.print_trainable_parameters()
    print(f"  (includes {asg_params:,} ASG parameters)")
    return model
