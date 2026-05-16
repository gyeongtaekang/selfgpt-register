"""
Training script: LlamaWithRegisters on CNN/DailyMail (max_length=1024).

DDP:
  torchrun --nproc_per_node=3 -m register_llama.train_cnn

Single GPU:
  python -m register_llama.train_cnn
"""

import os, sys, argparse, json, torch
from dataclasses import asdict
from transformers import (
    TrainingArguments, Trainer,
    TrainerCallback, TrainerState, TrainerControl, set_seed,
)

sys.path.insert(0, "/home/gyeongtae/selfcheckgpt")

from register_llama.config import Config
from register_llama.model import load_tokenizer, load_model, apply_lora, get_register_ids
from register_llama.dataset import CNNDailyMailDataset


def get_local_rank(): return int(os.environ.get("LOCAL_RANK", 0))
def is_main():        return get_local_rank() == 0


class LossLogCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not is_main() or logs is None:
            return
        entry = {"step": state.global_step}
        entry.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        self.history.append(entry)
        with open(os.path.join(self.output_dir, "loss_history_cnn.json"), "w") as f:
            json.dump(self.history, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_epochs",                  type=int,   default=None)
    p.add_argument("--per_device_train_batch_size", type=int,   default=None)
    p.add_argument("--gradient_accumulation_steps", type=int,   default=None)
    p.add_argument("--learning_rate",               type=float, default=None)
    p.add_argument("--output_dir",                  type=str,   default=None)
    p.add_argument("--lora_r",                      type=int,   default=None)
    p.add_argument("--max_length",                  type=int,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    # CNN/DailyMail overrides
    cfg.output_dir                  = cfg.cnn_output_dir
    cfg.max_length                  = cfg.cnn_max_length
    cfg.per_device_train_batch_size = cfg.cnn_per_device_train_batch_size
    cfg.gradient_accumulation_steps = cfg.cnn_gradient_accumulation_steps

    # CLI overrides
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    if is_main():
        print("=" * 60)
        print("LlamaWithRegisters (CNN/DailyMail) — DDP Training")
        world = int(os.environ.get("WORLD_SIZE", 1))
        eff_bs = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world
        print(f"GPUs: {world}  |  per-device bs: {cfg.per_device_train_batch_size}"
              f"  |  grad_accum: {cfg.gradient_accumulation_steps}"
              f"  |  effective bs: {eff_bs}")
        print(f"max_length: {cfg.max_length}")
        print(json.dumps(asdict(cfg), indent=2))
        print("=" * 60)

    tokenizer = load_tokenizer(cfg)
    model, orig_vocab = load_model(cfg, tokenizer, device_map={"": local_rank})

    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model = apply_lora(model, cfg, orig_vocab, tokenizer)

    register_ids = get_register_ids(tokenizer, cfg)
    if is_main():
        print(f"Register IDs: {register_ids}")

    train_dataset = CNNDailyMailDataset("train", cfg, tokenizer, register_ids)
    val_dataset   = CNNDailyMailDataset("validation", cfg, tokenizer, register_ids)

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
        eval_dataset=val_dataset,
        callbacks=[LossLogCallback(cfg.output_dir)],
    )

    if is_main():
        print("\nStarting CNN/DailyMail training …")
    trainer.train()

    if is_main():
        final_dir = os.path.join(cfg.output_dir, "final")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "train_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        print(f"\nModel saved to {final_dir}")


if __name__ == "__main__":
    main()
