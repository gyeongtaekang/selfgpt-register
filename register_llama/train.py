"""
Training script: LLaMA 3.1 8B + Register Tokens (LoRA fine-tuning on ROCStories).

DDP (권장 — VRAM 최대 활용):
  torchrun --nproc_per_node=3 -m register_llama.train

Single GPU:
  python -m register_llama.train
"""

import os
import sys
import argparse
import json
import torch
from dataclasses import asdict
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
)

sys.path.insert(0, "/home/gyeongtae/selfcheckgpt")

from register_llama.config import Config
from register_llama.model import (
    load_tokenizer, load_model, apply_lora, get_register_ids
)
from register_llama.dataset import (
    ROCStoriesDataset, ClozeTestDataset, evaluate_cloze_accuracy
)


# ── DDP helpers ────────────────────────────────────────────────────────────

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process() -> bool:
    return get_local_rank() == 0


# ── Cloze callback ─────────────────────────────────────────────────────────

class ClozeEvalCallback(TrainerCallback):
    """Evaluates Story Cloze accuracy on rank-0 only."""

    def __init__(self, model, tokenizer, register_ids, val_dataset, cfg: Config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.register_ids = register_ids
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.device = device
        self.history = []

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only rank-0 evaluates to avoid redundant work in DDP
        if not is_main_process():
            return

        acc, n = evaluate_cloze_accuracy(
            self.model, self.tokenizer, self.register_ids,
            self.val_dataset, self.device, self.cfg.max_length,
        )
        print(f"\n[Cloze Eval] step={state.global_step}  acc={acc:.4f}  n={n}")
        self.history.append({"step": state.global_step, "cloze_acc": acc})

        if is_main_process():
            hist_path = os.path.join(self.cfg.output_dir, "cloze_history.json")
            with open(hist_path, "w") as f:
                json.dump(self.history, f, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    if is_main_process():
        print("=" * 60)
        print("Register-Token LLaMA  ─  DDP Training")
        world = int(os.environ.get("WORLD_SIZE", 1))
        eff_bs = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world
        print(f"GPUs: {world}  |  per-device bs: {cfg.per_device_train_batch_size}"
              f"  |  grad_accum: {cfg.gradient_accumulation_steps}"
              f"  |  effective bs: {eff_bs}")
        print(json.dumps(asdict(cfg), indent=2))
        print("=" * 60)

    # ── Model: load onto this process's GPU (DDP-safe, no device_map splitting) ──
    tokenizer = load_tokenizer(cfg)

    # device_map={"": local_rank} → full model on one GPU per process
    model, orig_vocab_size = load_model(
        cfg, tokenizer, device_map={"": local_rank}
    )

    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model = apply_lora(model, cfg, orig_vocab_size, tokenizer)

    register_ids = get_register_ids(tokenizer, cfg)
    if is_main_process():
        print(f"Register IDs: {register_ids}")

    # ── Datasets ───────────────────────────────────────────────────────────
    train_dataset = ROCStoriesDataset(cfg, tokenizer, register_ids)
    val_dataset = ClozeTestDataset(cfg.val_dir, tokenizer, register_ids, cfg.max_length)

    # ── TrainingArguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        fp16=False,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,
        load_best_model_at_end=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        seed=cfg.seed,
        ddp_find_unused_parameters=False,
    )

    cloze_callback = ClozeEvalCallback(
        model, tokenizer, register_ids, val_dataset, cfg, device
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # placeholder; real eval via callback
        callbacks=[cloze_callback],
    )

    if is_main_process():
        print("\nStarting training …")
    trainer.train()

    # ── Save (rank-0 only) ────────────────────────────────────────────────
    if is_main_process():
        final_dir = os.path.join(cfg.output_dir, "final")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "train_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        print(f"\nModel saved to {final_dir}")

        acc, n = evaluate_cloze_accuracy(
            model, tokenizer, register_ids, val_dataset, device,
            cfg.max_length, verbose=True
        )
        print(f"\nFinal Validation Cloze Accuracy: {acc:.4f} (n={n})")


if __name__ == "__main__":
    main()
