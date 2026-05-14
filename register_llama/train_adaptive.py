"""
Training script: LlamaWithAdaptiveRegisters (position-0 registers + per-layer ASG).

DDP:
  torchrun --nproc_per_node=3 -m register_llama.train_adaptive

Single GPU:
  python -m register_llama.train_adaptive
"""

import os, sys, argparse, json, torch
from dataclasses import asdict
from transformers import (
    TrainingArguments, Trainer,
    TrainerCallback, TrainerState, TrainerControl, set_seed,
)

sys.path.insert(0, "/home/gyeongtae/selfcheckgpt")

from register_llama.config import Config
from register_llama.model import load_tokenizer, get_register_ids
from register_llama.adaptive_model import load_adaptive_model, apply_lora_adaptive
from register_llama.dataset import ROCStoriesDataset, ClozeTestDataset, evaluate_cloze_accuracy


def get_local_rank(): return int(os.environ.get("LOCAL_RANK", 0))
def is_main():        return get_local_rank() == 0


class ClozeCallback(TrainerCallback):
    def __init__(self, model, tokenizer, register_ids, val_dataset, cfg, device):
        self.model, self.tokenizer = model, tokenizer
        self.register_ids, self.val_dataset = register_ids, val_dataset
        self.cfg, self.device = cfg, device
        self.history = []

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not is_main(): return
        acc, n = evaluate_cloze_accuracy(
            self.model, self.tokenizer, self.register_ids,
            self.val_dataset, self.device, self.cfg.max_length,
        )
        print(f"\n[Cloze Eval] step={state.global_step}  acc={acc:.4f}  n={n}")
        self.history.append({"step": state.global_step, "cloze_acc": acc})
        with open(os.path.join(self.cfg.output_dir, "cloze_history_adaptive.json"), "w") as f:
            json.dump(self.history, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_epochs",                  type=int,   default=None)
    p.add_argument("--per_device_train_batch_size", type=int,   default=None)
    p.add_argument("--learning_rate",               type=float, default=None)
    p.add_argument("--output_dir",                  type=str,   default=None)
    p.add_argument("--lora_r",                      type=int,   default=None)
    p.add_argument("--asg_z_threshold",             type=float, default=None)
    p.add_argument("--asg_max_alpha",               type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    # Override defaults from CLI
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    # Save to a separate output dir to avoid overwriting base-model checkpoints
    if cfg.output_dir == "/home/gyeongtae/selfcheckgpt/checkpoints":
        cfg.output_dir = "/home/gyeongtae/selfcheckgpt/checkpoints_adaptive"

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    if is_main():
        print("=" * 60)
        print("LlamaWithAdaptiveRegisters — DDP Training")
        world = int(os.environ.get("WORLD_SIZE", 1))
        eff_bs = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world
        print(f"GPUs: {world}  |  per-device bs: {cfg.per_device_train_batch_size}"
              f"  |  grad_accum: {cfg.gradient_accumulation_steps}"
              f"  |  effective bs: {eff_bs}")
        print(f"ASG z_threshold={cfg.asg_z_threshold}  max_alpha={cfg.asg_max_alpha}")
        print(json.dumps(asdict(cfg), indent=2))
        print("=" * 60)

    tokenizer = load_tokenizer(cfg)
    model, orig_vocab = load_adaptive_model(cfg, tokenizer, device_map={"": local_rank})

    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model = apply_lora_adaptive(model, cfg, orig_vocab, tokenizer)

    register_ids = get_register_ids(tokenizer, cfg)
    if is_main():
        print(f"Register IDs: {register_ids}")

    train_dataset = ROCStoriesDataset(cfg, tokenizer, register_ids)
    val_dataset   = ClozeTestDataset(cfg.val_dir, tokenizer, register_ids, cfg.max_length)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        callbacks=[ClozeCallback(model, tokenizer, register_ids, val_dataset, cfg, device)],
    )

    if is_main():
        print("\nStarting adaptive training …")
    trainer.train()

    if is_main():
        final_dir = os.path.join(cfg.output_dir, "final")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "train_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        print(f"\nModel saved to {final_dir}")

        acc, n = evaluate_cloze_accuracy(
            model, tokenizer, register_ids, val_dataset, device, cfg.max_length, verbose=True
        )
        print(f"\nFinal Validation Cloze Accuracy: {acc:.4f} (n={n})")


if __name__ == "__main__":
    main()
