"""
Baseline: plain LLaMA 3.1 8B (no register tokens, no fine-tuning)
Reuses evaluate.py run_selfcheck with register_ids=[] for identical code path.
"""

import os, sys, json, random
import torch

sys.path.insert(0, "/home/gyeongtae/selfcheckgpt")

from transformers import LlamaForCausalLM, AutoTokenizer
from register_llama.config import Config
from register_llama.dataset import ClozeTestDataset
from register_llama.evaluate import run_selfcheck

SEED = 42
random.seed(SEED)
cfg = Config()

# ── 베이스라인 모델 로드 ────────────────────────────────────────────────
print("Loading LLaMA 3.1 8B (no register, no fine-tuning)...")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    cfg.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
device = torch.device("cuda:0")

# ── 데이터셋 (register_ids=[] → 레지스터 삽입 없음) ──────────────────
test_dataset = ClozeTestDataset(cfg.test_dir, tokenizer, [], cfg.max_length)

# ── evaluate.py의 run_selfcheck 동일하게 사용 ─────────────────────────
baseline = run_selfcheck(model, tokenizer, [], test_dataset, cfg, device)

# ── 기존 Register-LoRA 결과와 비교 ────────────────────────────────────
with open("results/results.json") as f:
    reg = json.load(f)["selfcheck"]

comparison = {
    "seed": SEED,
    "n_examples": cfg.num_eval_examples,
    "baseline_llama": baseline,
    "register_lora": {k: reg[k] for k in ["mean","std","median","min","max"]},
    "delta_mean": reg["mean"] - baseline["mean"],
}
os.makedirs("results", exist_ok=True)
with open("results/comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("\n" + "="*60)
print("  COMPARISON: Baseline LLaMA vs Register-LoRA")
print("="*60)
print(f"{'Metric':<18} {'Baseline':>12} {'Reg-LoRA':>12} {'Delta':>10}")
print("-"*56)
for k in ["mean", "std", "median", "min", "max"]:
    b = baseline[k]
    r = reg[k]
    print(f"{k:<18} {b:>12.4f} {r:>12.4f} {r-b:>+10.4f}")
print("="*60)
delta = comparison["delta_mean"]
print(f"  mean delta: {delta:+.4f}  ({'할루시네이션 감소 ✓' if delta < 0 else '증가 또는 동일'})")
print(f"\nSaved → results/comparison.json")
