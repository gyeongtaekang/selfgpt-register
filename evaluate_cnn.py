"""
SelfCheckGPT evaluation on CNN/DailyMail for:
  1. Baseline LLaMA (no register, no fine-tuning)
  2. Register-LoRA fine-tuned on CNN/DailyMail
  3. (optional) LoRA-only fine-tuned on CNN/DailyMail

Run after train_cnn.py completes:
  python -m evaluate_cnn --checkpoint checkpoints_cnn/final
"""

import os, sys, json, random, argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/home/gyeongtae/selfcheckgpt")

from transformers import LlamaForCausalLM, AutoTokenizer
from register_llama.config import Config
from register_llama.model import load_tokenizer, get_register_ids
from register_llama.dataset import CNNDailyMailEvalDataset
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

SEED = 42
random.seed(SEED)


@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    register_ids,
    prompt_text: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[str]:
    """Generate n_samples continuations for a prompt."""
    # Build input with register tokens
    bos = tokenizer.bos_token_id
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = [bos] + register_ids + prompt_ids
    input_tensor = torch.tensor([input_ids], device=device)

    samples = []
    model.eval()
    for _ in range(n_samples):
        out = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        new_tokens = out[0][len(input_ids):]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        samples.append(text)
    return samples


def run_selfcheck_cnn(model, tokenizer, register_ids, eval_dataset, cfg, device):
    """Run SelfCheckGPT on CNN/DailyMail eval set, return score dict."""
    checker = SelfCheckBERTScore(rescale_with_baseline=True)
    scores = []

    for item in tqdm(eval_dataset.items, desc="SelfCheckGPT"):
        prompt = item["prompt"]

        samples = generate_samples(
            model, tokenizer, register_ids, prompt,
            cfg.num_selfcheck_samples, cfg.max_new_tokens,
            cfg.temperature, cfg.top_p, device,
        )

        # First sample is the "passage", rest are for consistency check
        if not samples or not samples[0].strip():
            continue
        passage = samples[0]
        other_samples = samples[1:] if len(samples) > 1 else samples

        try:
            sent_scores = checker.predict(
                sentences=passage.split(". "),
                sampled_passages=other_samples,
            )
            if sent_scores:
                scores.append(float(np.mean(sent_scores)))
        except Exception:
            continue

    arr = np.array(scores) if scores else np.array([0.0])
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "median": float(np.median(arr)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "n":      len(scores),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to fine-tuned checkpoint (e.g. checkpoints_cnn/final). "
                        "If omitted, only baseline is evaluated.")
    p.add_argument("--lora_only_checkpoint", type=str, default=None,
                   help="Optional LoRA-only checkpoint for three-way comparison.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.max_length = cfg.cnn_max_length
    device = torch.device("cuda:0")

    eval_dataset = CNNDailyMailEvalDataset(
        split="test",
        tokenizer=AutoTokenizer.from_pretrained(cfg.model_path),
        register_ids=[],  # placeholder, will rebuild per model
        n_examples=cfg.num_eval_examples,
        prompt_tokens=cfg.cnn_prompt_tokens,
        seed=SEED,
    )

    results = {"seed": SEED, "n_examples": cfg.num_eval_examples}

    # ── 1. Baseline LLaMA ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating: Baseline LLaMA 3.1 8B (no register, no fine-tuning)")
    base_tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = LlamaForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # Rebuild eval_dataset with empty register_ids
    eval_dataset_base = CNNDailyMailEvalDataset(
        split="test", tokenizer=base_tokenizer, register_ids=[],
        n_examples=cfg.num_eval_examples, prompt_tokens=cfg.cnn_prompt_tokens, seed=SEED,
    )
    results["baseline_llama"] = run_selfcheck_cnn(base_model, base_tokenizer, [], eval_dataset_base, cfg, device)
    del base_model
    torch.cuda.empty_cache()

    # ── 2. Register-LoRA CNN fine-tuned ───────────────────────────────────
    if args.checkpoint:
        print("\n" + "=" * 60)
        print(f"Evaluating: Register-LoRA (checkpoint={args.checkpoint})")
        from register_llama.model import load_trained_model
        reg_tokenizer = load_tokenizer(cfg)
        register_ids = get_register_ids(reg_tokenizer, cfg)
        reg_model = load_trained_model(args.checkpoint, cfg, reg_tokenizer)
        eval_dataset_reg = CNNDailyMailEvalDataset(
            split="test", tokenizer=reg_tokenizer, register_ids=register_ids,
            n_examples=cfg.num_eval_examples, prompt_tokens=cfg.cnn_prompt_tokens, seed=SEED,
        )
        results["register_lora_cnn"] = run_selfcheck_cnn(
            reg_model, reg_tokenizer, register_ids, eval_dataset_reg, cfg, device
        )
        del reg_model
        torch.cuda.empty_cache()

    # ── 3. LoRA-only CNN fine-tuned (optional) ────────────────────────────
    if args.lora_only_checkpoint:
        print("\n" + "=" * 60)
        print(f"Evaluating: LoRA-only (checkpoint={args.lora_only_checkpoint})")
        from register_llama.model import load_trained_model
        lo_tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        lo_tokenizer.pad_token = lo_tokenizer.eos_token
        lo_model = load_trained_model(args.lora_only_checkpoint, cfg, lo_tokenizer)
        eval_dataset_lo = CNNDailyMailEvalDataset(
            split="test", tokenizer=lo_tokenizer, register_ids=[],
            n_examples=cfg.num_eval_examples, prompt_tokens=cfg.cnn_prompt_tokens, seed=SEED,
        )
        results["lora_only_cnn"] = run_selfcheck_cnn(
            lo_model, lo_tokenizer, [], eval_dataset_lo, cfg, device
        )
        del lo_model
        torch.cuda.empty_cache()

    # ── Save & print ───────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/cnn_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("  CNN/DailyMail SelfCheckGPT Results")
    print("=" * 60)
    models = [k for k in results if k not in ("seed", "n_examples")]
    header = f"{'Metric':<12}" + "".join(f"{m:>22}" for m in models)
    print(header)
    print("-" * len(header))
    for metric in ["mean", "std", "median", "min", "max", "n"]:
        row = f"{metric:<12}"
        for m in models:
            val = results[m].get(metric, float("nan"))
            row += f"{val:>22.4f}"
        print(row)
    print("=" * 60)

    if "baseline_llama" in results and "register_lora_cnn" in results:
        delta = results["register_lora_cnn"]["mean"] - results["baseline_llama"]["mean"]
        print(f"\n  Register-LoRA vs Baseline delta_mean: {delta:+.4f} "
              f"({'hallucination 감소 ✓' if delta < 0 else '증가 또는 동일'})")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
