"""
LlamaWithRegisters: LLaMA 3.1 + register tokens at position 0.

Token layout:
  [BOS] [REG1] [REG2] [REG3] [REG4] [t1] [t2] [t3] ...
pos: 0     0     0     0     0    1    2    3

Registers share position 0 with BOS → absorb attention sinks without
consuming positional information from content tokens.
"""

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, TaskType

from register_llama.config import Config


def _chunked_cross_entropy(
    logits: torch.Tensor,    # (N, V) bf16
    labels: torch.Tensor,    # (N,)   long
    chunk_size: int = 512,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute CE loss in fp32 chunks to avoid allocating a full fp32 logits copy.
    Peak extra memory: chunk_size × vocab × 4 bytes (~262 MB for chunk=512, V=128K).
    """
    total_loss = logits.new_zeros(())
    n_valid = 0
    for start in range(0, logits.size(0), chunk_size):
        end = min(start + chunk_size, logits.size(0))
        chunk_logits = logits[start:end].float()       # only this slice to fp32
        chunk_labels = labels[start:end]
        mask = chunk_labels != ignore_index
        if not mask.any():
            continue
        chunk_loss = F.cross_entropy(
            chunk_logits, chunk_labels, ignore_index=ignore_index, reduction="sum"
        )
        total_loss = total_loss + chunk_loss
        n_valid += mask.sum()
    return total_loss / n_valid.clamp(min=1)


class LlamaWithRegisters(LlamaForCausalLM):
    """LlamaForCausalLM with register-token position override."""

    def __init__(self, config):
        super().__init__(config)
        self.num_registers = getattr(config, "num_registers", 4)

    # ── Position ID construction ────────────────────────────────────────────

    def _make_position_ids(self, seq_len: int, device: torch.device, batch_size: int) -> torch.Tensor:
        """
        [0, 0, …, 0, 1, 2, 3, …]
         └──n_prefix──┘ └─content─┘
        """
        n = 1 + self.num_registers  # BOS + REGs at position 0
        pos = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        if seq_len > n:
            pos[:, n:] = torch.arange(1, seq_len - n + 1, device=device)
        return pos

    def _get_past_len(self, past_key_values) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):   # DynamicCache
            return past_key_values.get_seq_length()
        if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
            layer = past_key_values[0]
            if isinstance(layer, (list, tuple)) and len(layer) > 0:
                return layer[0].shape[2]
        return 0

    # ── Forward override ───────────────────────────────────────────────────

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        labels=None,
        **kwargs,
    ):
        # ── Build register-aware position IDs ──────────────────────────────
        if position_ids is None and input_ids is not None:
            bs, L = input_ids.shape
            dev = input_ids.device
            past_len = self._get_past_len(past_key_values)

            if past_len == 0:
                position_ids = self._make_position_ids(L, dev, bs)
            else:
                n_prefix = 1 + self.num_registers
                base = max(1, past_len - n_prefix + 1)
                position_ids = torch.arange(
                    base, base + L, device=dev
                ).unsqueeze(0).expand(bs, -1)

        # ── Run backbone without labels to avoid the .float() logit cast ──
        # transformers 5.x casts logits to float32 before CE loss, which
        # doubles the logits memory (e.g. 131 MB bf16 → 262 MB fp32).
        # We compute CE on bf16 logits ourselves to save ~130 MB per step.
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            labels=None,   # skip internal loss computation
            **kwargs,
        )

        loss = None
        if labels is not None:
            logits = outputs.logits                   # bf16, shape (B, L, V)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            # Chunked CE: cast to fp32 in small chunks to avoid materialising
            # a full fp32 logits tensor (e.g. bs=16 × 256 × 128K = 2 GB).
            loss = _chunked_cross_entropy(
                shift_logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
                chunk_size=128,    # 128 × 128K × 4B = 65 MB per chunk
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ── Setup helpers ───────────────────────────────────────────────────────────

def load_tokenizer(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Add register special tokens
    num_added = tok.add_special_tokens(
        {"additional_special_tokens": cfg.register_tokens}
    )
    print(f"Added {num_added} register tokens → vocab size: {len(tok)}")
    return tok


def load_model(cfg: Config, tokenizer, device_map="auto"):
    # Patch config with num_registers so it persists in saved checkpoints
    model_cfg = AutoConfig.from_pretrained(cfg.model_path)
    model_cfg.num_registers = cfg.num_registers

    model = LlamaWithRegisters.from_pretrained(
        cfg.model_path,
        config=model_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    # Expand embedding table for new tokens
    orig_vocab_size = model_cfg.vocab_size
    model.resize_token_embeddings(len(tokenizer))

    # Initialise register embeddings near BOS embedding
    _init_register_embeddings(model, tokenizer, cfg, orig_vocab_size)

    return model, orig_vocab_size


def _init_register_embeddings(model, tokenizer, cfg: Config, orig_vocab_size: int):
    embed = model.get_input_embeddings()
    bos_id = tokenizer.bos_token_id
    bos_vec = embed.weight.data[bos_id].clone()

    for token in cfg.register_tokens:
        rid = tokenizer.convert_tokens_to_ids(token)
        # Small noise so each register starts differently
        embed.weight.data[rid] = bos_vec + 0.01 * torch.randn_like(bos_vec)

    print(f"Register embeddings initialised from BOS (id={bos_id})")


def apply_lora(model, cfg: Config, orig_vocab_size: int, tokenizer):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Freeze the embedding entirely — training it requires a full float32
    # gradient tensor (128K × 4096 × 4B = 2.1 GB) which causes OOM.
    # Register embeddings are pre-initialised from BOS and kept fixed;
    # the LoRA adapters on attention layers learn to use them effectively.
    model.get_input_embeddings().weight.requires_grad_(False)
    print("Embedding frozen (register tokens initialised from BOS, no grad needed).")
    return model


def get_register_ids(tokenizer, cfg: Config):
    return [tokenizer.convert_tokens_to_ids(t) for t in cfg.register_tokens]
