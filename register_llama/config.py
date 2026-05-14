from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    model_path: str = "/home/gyeongtae/models/llama31-8b"
    num_registers: int = 4
    register_tokens: List[str] = field(
        default_factory=lambda: ["<|REG1|>", "<|REG2|>", "<|REG3|>", "<|REG4|>"]
    )

    # ── Data ───────────────────────────────────────────────────────────────
    train_dir: str = "/home/gyeongtae/selfcheckgpt/train"
    val_dir: str = "/home/gyeongtae/selfcheckgpt/val"
    test_dir: str = "/home/gyeongtae/selfcheckgpt/test"
    max_length: int = 256

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # ── Training ───────────────────────────────────────────────────────────
    output_dir: str = "/home/gyeongtae/selfcheckgpt/checkpoints"
    num_epochs: int = 3
    per_device_train_batch_size: int = 16  # DDP: 3 GPU × 16 × 2 = effective 96
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    eval_steps: int = 200
    save_steps: int = 200
    bf16: bool = True
    gradient_checkpointing: bool = True    # saves activation mem → allows large batch
    seed: int = 42

    # ── Adaptive Sink Gate (ASG) ───────────────────────────────────────────
    num_asg_absorbers: int = 4      # learnable absorber directions per layer
    asg_z_threshold: float = 2.0   # initial z-score cutoff (learnable during training)
    asg_max_alpha: float = 0.5     # max fraction of sink component removed (hard cap)

    # ── Evaluation ─────────────────────────────────────────────────────────
    num_selfcheck_samples: int = 5    # generation samples per prompt
    num_eval_examples: int = 50       # test examples for SelfCheckGPT
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_p: float = 0.9

    # ── Report ─────────────────────────────────────────────────────────────
    report_path: str = "/home/gyeongtae/selfcheckgpt/results/report.md"
