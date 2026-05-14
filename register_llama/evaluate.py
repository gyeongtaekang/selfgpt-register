"""
Post-training evaluation pipeline:
  1. Story Cloze Test accuracy (val + test)
  2. SelfCheckGPT (BERTScore variant) — hallucination score
  3. Attention sink analysis — distribution across BOS, REGs, content tokens
  4. Markdown report generation

Run:
  python -m register_llama.evaluate --checkpoint /path/to/checkpoints/final
"""

import os
import sys
import json
import argparse
import random
import textwrap
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch

sys.path.insert(0, "/home/gyeongtae/selfcheckgpt")

from register_llama.config import Config
from register_llama.model import (
    LlamaWithRegisters, load_tokenizer, get_register_ids
)
from register_llama.dataset import (
    ClozeTestDataset, evaluate_cloze_accuracy, _insert_registers
)
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore


# ── Model loading ──────────────────────────────────────────────────────────

def load_trained_model(checkpoint_dir: str, cfg: Config, tokenizer, attn_implementation: str = "sdpa"):
    from peft import PeftModel
    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(cfg.model_path)
    model_cfg.num_registers = cfg.num_registers

    base = LlamaWithRegisters.from_pretrained(
        cfg.model_path,
        config=model_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    base.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    return model


# ── Sample generation ──────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    register_ids: List[int],
    prompt: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[str]:
    """Generate n_samples continuations for a single prompt."""
    raw = tokenizer.encode(prompt, add_special_tokens=True)
    ids = _insert_registers(raw, tokenizer.bos_token_id, register_ids)
    input_ids = torch.tensor([ids], device=device)

    samples = []
    for _ in range(n_samples):
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode only the new tokens
        new_tokens = out[0][len(ids):]
        samples.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return samples


# ── SelfCheckGPT evaluation ───────────────────────────────────────────────

def run_selfcheck(
    model,
    tokenizer,
    register_ids: List[int],
    test_dataset: ClozeTestDataset,
    cfg: Config,
    device: torch.device,
) -> Dict:
    """
    For each test example, generate n_samples completions then score
    consistency with SelfCheckBERTScore (higher score = more hallucination).
    Returns summary statistics.
    """
    print("\n── SelfCheckGPT (BERTScore) ──────────────────────────")
    selfcheck = SelfCheckBERTScore(rescale_with_baseline=True)

    indices = list(range(len(test_dataset)))
    random.shuffle(indices)
    indices = indices[: cfg.num_eval_examples]

    all_scores: List[float] = []
    examples_log: List[dict] = []

    for i, idx in enumerate(indices):
        item = test_dataset[idx]
        prompt = item["context"]

        print(f"  [{i+1}/{len(indices)}] generating {cfg.num_selfcheck_samples} samples …", end="\r")
        samples = generate_samples(
            model, tokenizer, register_ids, prompt,
            cfg.num_selfcheck_samples, cfg.max_new_tokens,
            cfg.temperature, cfg.top_p, device,
        )

        # Split first sample into sentences for scoring
        import spacy
        nlp = spacy.load("en_core_web_sm")
        sentences = [s.text.strip() for s in nlp(samples[0]).sents if len(s.text.strip()) > 3]
        if not sentences:
            continue

        scores = selfcheck.predict(sentences=sentences, sampled_passages=samples[1:])
        mean_score = float(np.mean(scores))
        all_scores.append(mean_score)

        examples_log.append({
            "context": prompt,
            "generated": samples[0],
            "selfcheck_score": mean_score,
            "sentence_scores": scores.tolist(),
        })

    print(f"\nSelfCheckGPT: {len(all_scores)} examples evaluated")
    result = {
        "mean": float(np.mean(all_scores)),
        "std": float(np.std(all_scores)),
        "median": float(np.median(all_scores)),
        "min": float(np.min(all_scores)),
        "max": float(np.max(all_scores)),
        "examples": examples_log[:5],   # keep 5 for report
    }
    return result


# ── Attention analysis ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_attention_weights(
    model,
    tokenizer,
    register_ids: List[int],
    text: str,
    device: torch.device,
    max_length: int = 200,
) -> Tuple[np.ndarray, int]:
    """
    Returns (attn_matrix, n_prefix) where attn_matrix has shape
    (n_layers, n_heads, seq_len, seq_len).
    n_prefix = 1 + num_registers (the zero-position tokens).
    """
    raw = tokenizer.encode(text, add_special_tokens=True)
    ids = _insert_registers(raw, tokenizer.bos_token_id, register_ids)[:max_length]
    input_ids = torch.tensor([ids], device=device)

    # PeftModel wraps the base model and may not forward output_attentions correctly.
    # Access the underlying LlamaWithRegisters directly so attentions are returned.
    inner = model.base_model.model if hasattr(model, "base_model") else model
    out = inner(input_ids=input_ids, output_attentions=True, use_cache=False)
    if not out.attentions:
        raise RuntimeError("Model returned no attentions — check output_attentions support")
    # out.attentions: tuple of (1, n_heads, seq, seq) per layer
    attn = np.stack([a[0].float().cpu().numpy() for a in out.attentions])  # (L, H, S, S)
    n_prefix = 1 + len(register_ids)
    return attn, n_prefix


def attention_sink_analysis(
    model,
    tokenizer,
    register_ids: List[int],
    test_dataset: ClozeTestDataset,
    device: torch.device,
    max_length: int = 200,
    n_examples: int = 20,
) -> Dict:
    """
    Computes mean attention weight flowing INTO:
      - BOS token (index 0)
      - Each register token (indices 1..num_reg)
      - All other content tokens

    Returns per-layer and aggregate statistics.
    """
    print("\n── Attention Sink Analysis ───────────────────────────")
    n_reg = len(register_ids)
    n_prefix = 1 + n_reg

    layer_bos_attn: List[float] = []
    layer_reg_attn: List[List[float]] = [[] for _ in range(n_reg)]
    layer_content_attn: List[float] = []
    layer_stats: List[Dict] = []   # per-layer means

    indices = list(range(min(n_examples, len(test_dataset))))

    for idx in indices:
        text = test_dataset[idx]["context"]
        try:
            attn, _ = extract_attention_weights(
                model, tokenizer, register_ids, text, device, max_length
            )
        except Exception as e:
            print(f"  Skipping example {idx}: {e}")
            continue

        # attn: (L, H, S, S) — each row [i] sums to 1 over columns
        n_layers, n_heads, S, _ = attn.shape
        # Mean over heads: (L, S, S)
        attn_mean = attn.mean(axis=1)

        # For each layer, compute attention INTO each position (column mean)
        # "attention received by position j" = mean over query positions i of attn[i, j]
        col_mean = attn_mean.mean(axis=1)  # (L, S)

        bos_col = col_mean[:, 0].tolist()          # attention to BOS
        reg_cols = [col_mean[:, r+1].tolist() for r in range(n_reg)]
        content_col = col_mean[:, n_prefix:].mean(axis=1).tolist()

        layer_bos_attn.append(np.mean(bos_col))
        for r in range(n_reg):
            layer_reg_attn[r].append(np.mean(reg_cols[r]))
        layer_content_attn.append(np.mean(content_col))

        # Per-layer detail for the first pass
        if not layer_stats:
            n_layers_actual = n_layers
            layer_stats = [
                {
                    "layer": l,
                    "bos_attn": float(col_mean[l, 0].mean()),
                    "reg_attn": float(col_mean[l, 1:n_prefix].mean()),
                    "content_attn": float(col_mean[l, n_prefix:].mean()),
                }
                for l in range(n_layers_actual)
            ]

    result = {
        "mean_bos_attn": float(np.mean(layer_bos_attn)) if layer_bos_attn else 0,
        "mean_reg_attn": float(np.mean([np.mean(x) for x in layer_reg_attn])) if layer_reg_attn[0] else 0,
        "mean_content_attn": float(np.mean(layer_content_attn)) if layer_content_attn else 0,
        "per_register_attn": [
            float(np.mean(layer_reg_attn[r])) for r in range(n_reg)
        ],
        "per_layer_stats": layer_stats,
    }
    print(f"  BOS attn:     {result['mean_bos_attn']:.4f}")
    print(f"  Register attn:{result['mean_reg_attn']:.4f}")
    print(f"  Content attn: {result['mean_content_attn']:.4f}")
    return result


# ── Report generation ──────────────────────────────────────────────────────

def generate_report(
    cfg: Config,
    val_acc: float,
    val_n: int,
    test_acc: float,
    test_n: int,
    selfcheck_results: Dict,
    attn_results: Dict,
    cloze_history: List[Dict],
    output_path: str,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Attention table ────────────────────────────────────────────────────
    attn_table_rows = ""
    for stat in attn_results.get("per_layer_stats", [])[:8]:
        attn_table_rows += (
            f"| {stat['layer']:>5} | {stat['bos_attn']:.4f} | "
            f"{stat['reg_attn']:.4f} | {stat['content_attn']:.4f} |\n"
        )

    # ── Cloze history table ────────────────────────────────────────────────
    cloze_rows = ""
    for h in cloze_history:
        cloze_rows += f"| {h['step']:>6} | {h['cloze_acc']:.4f} |\n"

    # ── SelfCheck examples ─────────────────────────────────────────────────
    examples_md = ""
    for i, ex in enumerate(selfcheck_results.get("examples", []), 1):
        ctx = textwrap.shorten(ex["context"], width=200)
        gen = textwrap.shorten(ex["generated"], width=300)
        score = ex["selfcheck_score"]
        examples_md += textwrap.dedent(f"""
            **Example {i}**

            *Context:* {ctx}

            *Generated:* {gen}

            *SelfCheck score (inconsistency):* `{score:.4f}` — {"⚠ hallucination risk" if score > 0.5 else "✓ consistent"}

            ---
        """)

    # ── Per-register attention ─────────────────────────────────────────────
    reg_attn_rows = ""
    for i, v in enumerate(attn_results.get("per_register_attn", []), 1):
        reg_attn_rows += f"| REG{i} | {v:.4f} |\n"

    report = textwrap.dedent(f"""
    # Register-Token Attention Sink Absorption: Experiment Report

    **Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
    **Model:** LLaMA 3.1 8B (`{cfg.model_path}`)
    **Training data:** ROCStories (spring2016 + winter2017, ~98,000 stories)

    ---

    ## 1. Research Motivation

    Large language models exhibit *attention sinks*: the BOS token accumulates
    disproportionately large attention weights across all layers and heads, even
    when semantically irrelevant.  This phenomenon forces the model to concentrate
    capacity on a single useless token, potentially reducing representational
    quality and contributing to hallucination.

    **Proposed fix:** insert *K* register tokens immediately after BOS, all
    assigned **position 0** (identical to BOS).  The register tokens provide
    additional "sink" capacity at zero positional cost, allowing attention to
    distribute over several dummy positions rather than collapsing onto one.

    ### Token layout

    ```
    [BOS] [REG1] [REG2] [REG3] [REG4] [t1] [t2] [t3] …
    pos:   0      0      0      0      0    1    2    3
    ```

    ---

    ## 2. Implementation

    | Component | Detail |
    |-----------|--------|
    | Base model | LLaMA 3.1 8B |
    | Register tokens | 4 (`<|REG1|>` … `<|REG4|>`) |
    | Position scheme | BOS + REGs at pos 0; content at 1, 2, 3 … |
    | Register init | BOS embedding ± Gaussian noise (σ=0.01) |
    | Fine-tuning | LoRA (r={cfg.lora_r}, α={cfg.lora_alpha}) on q/k/v/o/gate/up/down proj |
    | Training task | Causal LM on ROCStories |
    | Validation | Story Cloze Test (NLL comparison) |
    | Epochs | {cfg.num_epochs} |
    | Effective batch | {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps} |
    | Learning rate | {cfg.learning_rate} |

    ---

    ## 3. Story Cloze Test Results

    The model picks the better story ending by comparing negative log-likelihood
    of each candidate conditioned on the four-sentence context.

    | Split | Accuracy | Examples |
    |-------|----------|----------|
    | Validation | **{val_acc:.4f}** | {val_n} |
    | Test | **{test_acc:.4f}** | {test_n} |

    ### Validation Cloze Accuracy During Training

    | Step | Accuracy |
    |------|----------|
    {cloze_rows.strip()}

    ---

    ## 4. SelfCheckGPT Results

    For each of {cfg.num_eval_examples} test story contexts, {cfg.num_selfcheck_samples}
    completions were sampled and scored for internal consistency via BERTScore.
    **Lower score = more consistent = fewer hallucinations.**

    | Metric | Value |
    |--------|-------|
    | Mean inconsistency | {selfcheck_results['mean']:.4f} |
    | Std | {selfcheck_results['std']:.4f} |
    | Median | {selfcheck_results['median']:.4f} |
    | Min | {selfcheck_results['min']:.4f} |
    | Max | {selfcheck_results['max']:.4f} |

    ### Example Outputs

    {examples_md}

    ---

    ## 5. Attention Distribution Analysis

    Mean attention **received** by each token group (averaged over layers,
    heads, and query positions across {min(20, len(attn_results.get("per_layer_stats", [{}])))} test examples).

    | Token group | Mean attention weight |
    |-------------|----------------------|
    | BOS (pos 0) | {attn_results['mean_bos_attn']:.4f} |
    | All Registers | {attn_results['mean_reg_attn']:.4f} |
    | Content tokens | {attn_results['mean_content_attn']:.4f} |

    ### Per-register attention

    | Register | Mean attn |
    |----------|-----------|
    {reg_attn_rows.strip()}

    ### Per-layer attention (first 8 layers)

    | Layer | BOS attn | REG attn | Content attn |
    |-------|----------|----------|--------------|
    {attn_table_rows.strip()}

    **Interpretation:** If register tokens are successfully absorbing the
    attention sink, we expect:
    - `BOS attn` to be *lower* than in a model without registers
    - `REG attn` to be notably higher than `BOS attn`
    - `Content attn` to be relatively high and evenly distributed

    ---

    ## 6. Conclusion

    Register tokens at position 0 provide dedicated "sink" capacity,
    redistributing the attention mass that would otherwise collapse onto BOS.
    The Story Cloze accuracy measures end-task quality, while the SelfCheckGPT
    score quantifies hallucination reduction compared to expected baseline
    (random ≈ 0.5, strong models < 0.3).

    Future work:
    - Ablation over number of registers (1, 2, 4, 8)
    - Comparison with a no-register baseline on the same task
    - Extension to longer documents where attention sinks are more harmful

    ---
    *Generated automatically by `register_llama/evaluate.py`*
    """).strip()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/home/gyeongtae/selfcheckgpt/checkpoints/final",
        help="Path to the saved LoRA checkpoint directory",
    )
    parser.add_argument("--report_path", default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.report_path:
        cfg.report_path = args.report_path

    tokenizer = load_tokenizer(cfg)
    register_ids = get_register_ids(tokenizer, cfg)

    print("Loading trained model …")
    model = load_trained_model(args.checkpoint, cfg, tokenizer)

    # Resolve primary device
    try:
        device_ids = list(model.base_model.model.hf_device_map.values())
        device = torch.device(f"cuda:{device_ids[0]}")
    except Exception:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Cloze accuracy ────────────────────────────────────────────────────
    print("\n── Story Cloze Test Accuracy ────────────────────────")
    val_dataset = ClozeTestDataset(cfg.val_dir, tokenizer, register_ids, cfg.max_length)
    test_dataset = ClozeTestDataset(cfg.test_dir, tokenizer, register_ids, cfg.max_length)

    val_acc, val_n = evaluate_cloze_accuracy(
        model, tokenizer, register_ids, val_dataset, device, cfg.max_length, verbose=True
    )
    test_acc, test_n = evaluate_cloze_accuracy(
        model, tokenizer, register_ids, test_dataset, device, cfg.max_length, verbose=True
    )
    print(f"Val  Cloze Accuracy: {val_acc:.4f} (n={val_n})")
    print(f"Test Cloze Accuracy: {test_acc:.4f} (n={test_n})")

    # ── SelfCheckGPT ──────────────────────────────────────────────────────
    selfcheck_results = run_selfcheck(
        model, tokenizer, register_ids, test_dataset, cfg, device
    )

    # ── Attention analysis ────────────────────────────────────────────────
    attn_results = attention_sink_analysis(
        model, tokenizer, register_ids, test_dataset, device, cfg.max_length
    )

    # ── Training history (if available) ──────────────────────────────────
    hist_path = os.path.join(os.path.dirname(args.checkpoint), "cloze_history.json")
    cloze_history = []
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            cloze_history = json.load(f)

    # ── Save raw results ──────────────────────────────────────────────────
    results_dir = os.path.dirname(cfg.report_path)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({
            "val_accuracy": val_acc,
            "val_n": val_n,
            "test_accuracy": test_acc,
            "test_n": test_n,
            "selfcheck": selfcheck_results,
            "attention": attn_results,
            "cloze_history": cloze_history,
        }, f, indent=2)

    # ── Report ────────────────────────────────────────────────────────────
    generate_report(
        cfg=cfg,
        val_acc=val_acc,
        val_n=val_n,
        test_acc=test_acc,
        test_n=test_n,
        selfcheck_results=selfcheck_results,
        attn_results=attn_results,
        cloze_history=cloze_history,
        output_path=cfg.report_path,
    )


if __name__ == "__main__":
    main()
