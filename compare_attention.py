"""
Attention map comparison using Attention-Viewer style (seaborn RdBu_r, causal mask).

Generates:
  assets/all_layers_avg_base.jpg         — Base LLaMA, all 32 layers
  assets/all_layers_avg_register.jpg     — Register+LoRA, all 32 layers
  assets/all_layers_avg_diff.jpg         — Difference per layer (base − register)
  assets/layer_{N}_comparison.jpg        — Side-by-side per-head grids for select layers

Run:
  python compare_attention.py
"""

import os, sys, math, torch
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, "/home/gyeongtae/selfcheckgpt")

from transformers import LlamaForCausalLM, AutoTokenizer
from register_llama.config import Config
from register_llama.model import load_tokenizer, get_register_ids
from register_llama.dataset import _insert_registers
from register_llama.evaluate import load_trained_model

CHECKPOINT = "/home/gyeongtae/selfcheckgpt/checkpoints/final"
ASSETS_DIR = "/home/gyeongtae/selfcheckgpt/Attention-Viewer/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

PROMPT = "Terry's daughter was in extreme pain in her mouth. After an evaluation, Terry realized that her daughter was teething. Once Terry realized she was teething, Terry administered medication."
MAX_LEN = 48
NUM_FIGS_PER_ROW = 4


# ── Attention extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def get_attn_base(model, tokenizer, text, device, max_len):
    ids = tokenizer.encode(text, add_special_tokens=True)[:max_len]
    tokens = [t.replace("▁", "").replace("Ġ", "") for t in
              tokenizer.convert_ids_to_tokens(ids)]
    input_ids = torch.tensor([ids], device=device)
    out = model(input_ids, output_attentions=True)
    # list of (1, H, S, S) tensors → keep on CPU
    attns = [a.detach().cpu() for a in out.attentions]
    return attns, tokens


@torch.no_grad()
def get_attn_register(model, tokenizer, register_ids, text, device, max_len):
    raw = tokenizer.encode(text, add_special_tokens=True)
    ids = _insert_registers(raw, tokenizer.bos_token_id, register_ids)[:max_len]
    n_reg = len(register_ids)
    raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    tokens = []
    for i, t in enumerate(raw_tokens):
        if i == 0:
            tokens.append("<s>")
        elif i <= n_reg:
            tokens.append(f"<R{i}>")
        else:
            tokens.append(t.replace("▁", "").replace("Ġ", ""))
    input_ids = torch.tensor([ids], device=device)
    inner = model.base_model.model if hasattr(model, "base_model") else model
    out = inner(input_ids=input_ids, output_attentions=True, use_cache=False)
    attns = [a.detach().cpu() for a in out.attentions]
    return attns, tokens, n_reg


# ── Plotting (Attention-Viewer style) ─────────────────────────────────────────

def _heatmap_layer(ax, attn_2d, tokens, title):
    """Draw one head-averaged causal attention heatmap on ax."""
    mask = torch.triu(torch.ones_like(attn_2d, dtype=torch.bool), diagonal=1)
    sns.heatmap(
        attn_2d.numpy(), mask=mask.numpy(),
        cmap="RdBu_r", square=True,
        xticklabels=tokens, yticklabels=tokens,
        ax=ax, cbar=False,
    )
    ax.set_title(title, fontsize=7)
    ax.tick_params(axis="both", labelsize=5)


def plot_all_layers(attns, tokens, suptitle, save_path, num_figs_per_row=4):
    """Replicate Attention-Viewer all_layers_avg style."""
    n_layers = len(attns)
    n_rows = math.ceil(n_layers / num_figs_per_row)
    S = len(tokens)
    fig, axes = plt.subplots(n_rows, num_figs_per_row,
                             figsize=(S * 1.6 * num_figs_per_row,
                                      S * 0.7 * n_rows))
    for layer_idx in tqdm(range(n_layers), desc=suptitle):
        row, col = layer_idx // num_figs_per_row, layer_idx % num_figs_per_row
        avg = attns[layer_idx][0].mean(dim=0).float()   # (S, S)
        _heatmap_layer(axes[row, col], avg, tokens, f"layer {layer_idx}")

    # hide unused axes
    for idx in range(n_layers, n_rows * num_figs_per_row):
        axes[idx // num_figs_per_row, idx % num_figs_per_row].axis("off")

    plt.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_diff_layers(attns_base, tokens_base, attns_reg, tokens_reg, n_reg,
                     save_path, num_figs_per_row=4):
    """
    Difference heatmap per layer: base − register (content tokens only).
    Strips BOS from base, strips BOS+REGs from register, aligns content tokens.
    """
    n_layers = len(attns_base)
    n_rows = math.ceil(n_layers / num_figs_per_row)

    # Content-only token labels (skip BOS for base, skip BOS+REGs for register)
    content_tokens = tokens_base[1:]   # same content tokens in both
    S = len(content_tokens)

    # Compute global vmax for symmetric colorscale
    diffs = []
    for l in range(n_layers):
        avg_base = attns_base[l][0].mean(dim=0).float()[1:, 1:].numpy()       # strip BOS
        avg_reg  = attns_reg[l][0].mean(dim=0).float()[1+n_reg:, 1+n_reg:].numpy()  # strip BOS+REGs
        min_S = min(avg_base.shape[0], avg_reg.shape[0], S)
        diffs.append(avg_base[:min_S, :min_S] - avg_reg[:min_S, :min_S])
    vmax = max(abs(d).max() for d in diffs)

    fig, axes = plt.subplots(n_rows, num_figs_per_row,
                             figsize=(S * 1.6 * num_figs_per_row,
                                      S * 0.7 * n_rows))
    for layer_idx in tqdm(range(n_layers), desc="Difference"):
        row, col = layer_idx // num_figs_per_row, layer_idx % num_figs_per_row
        diff = diffs[layer_idx]
        min_S = diff.shape[0]
        tok = content_tokens[:min_S]
        mask = np.triu(np.ones((min_S, min_S), dtype=bool), k=1)
        sns.heatmap(
            diff, mask=mask, cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            square=True,
            xticklabels=tok, yticklabels=tok,
            ax=axes[row, col], cbar=False,
        )
        axes[row, col].set_title(f"layer {layer_idx}", fontsize=7)
        axes[row, col].tick_params(axis="both", labelsize=5)

    for idx in range(n_layers, n_rows * num_figs_per_row):
        axes[idx // num_figs_per_row, idx % num_figs_per_row].axis("off")

    plt.suptitle("Attention Difference: Base LLaMA − Register+LoRA\n(Red = base attends more here)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_layer_comparison(attns_base, tokens_base, attns_reg, tokens_reg,
                          layer_idx, save_path, num_figs_per_row=4):
    """Side-by-side per-head grids for one layer (base top, register bottom)."""
    n_heads = attns_base[layer_idx].shape[1]
    n_rows = math.ceil(n_heads / num_figs_per_row)
    S_b = len(tokens_base)
    S_r = len(tokens_reg)

    fig, axes = plt.subplots(n_rows * 2, num_figs_per_row,
                             figsize=(max(S_b, S_r) * 1.4 * num_figs_per_row,
                                      max(S_b, S_r) * 0.6 * n_rows * 2))

    for head_idx in tqdm(range(n_heads), desc=f"Layer {layer_idx} heads"):
        row, col = head_idx // num_figs_per_row, head_idx % num_figs_per_row
        # Base
        h_base = attns_base[layer_idx][0, head_idx].float()
        _heatmap_layer(axes[row, col], h_base, tokens_base, f"Base head {head_idx}")
        # Register
        h_reg = attns_reg[layer_idx][0, head_idx].float()
        _heatmap_layer(axes[row + n_rows, col], h_reg, tokens_reg, f"Reg head {head_idx}")

    for idx in range(n_heads, n_rows * num_figs_per_row):
        axes[idx // num_figs_per_row, idx % num_figs_per_row].axis("off")
        axes[idx // num_figs_per_row + n_rows, idx % num_figs_per_row].axis("off")

    plt.suptitle(f"Layer {layer_idx} — Top: Base LLaMA  |  Bottom: Register+LoRA",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    device = torch.device("cuda:0")

    # ── Base LLaMA ────────────────────────────────────────────────────────────
    print("Loading Base LLaMA…")
    base_tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = LlamaForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager",
    ).eval()
    attns_base, tokens_base = get_attn_base(base_model, base_tokenizer, PROMPT, device, MAX_LEN)
    del base_model; torch.cuda.empty_cache()

    # ── Register+LoRA ─────────────────────────────────────────────────────────
    print("Loading Register+LoRA…")
    reg_tokenizer = load_tokenizer(cfg)
    register_ids = get_register_ids(reg_tokenizer, cfg)
    reg_model = load_trained_model(CHECKPOINT, cfg, reg_tokenizer, attn_implementation="eager").eval()
    attns_reg, tokens_reg, n_reg = get_attn_register(
        reg_model, reg_tokenizer, register_ids, PROMPT, device, MAX_LEN)
    del reg_model; torch.cuda.empty_cache()

    print(f"\nBase tokens: {len(tokens_base)}  |  Register tokens: {len(tokens_reg)}\n")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("[1/4] Base LLaMA — all layers avg…")
    plot_all_layers(attns_base, tokens_base,
                    "all layers avg — Base LLaMA",
                    os.path.join(ASSETS_DIR, "all_layers_avg_base.jpg"))

    print("[2/4] Register+LoRA — all layers avg…")
    plot_all_layers(attns_reg, tokens_reg,
                    "all layers avg — Register+LoRA (w/ 4 registers)",
                    os.path.join(ASSETS_DIR, "all_layers_avg_register.jpg"))

    print("[3/4] Difference — all layers…")
    plot_diff_layers(attns_base, tokens_base, attns_reg, tokens_reg, n_reg,
                     os.path.join(ASSETS_DIR, "all_layers_avg_diff.jpg"))

    print("[4/4] Layer 15 — per-head side-by-side…")
    plot_layer_comparison(attns_base, tokens_base, attns_reg, tokens_reg,
                          layer_idx=15,
                          save_path=os.path.join(ASSETS_DIR, "layer_15_comparison.jpg"))

    print(f"\nAll figures saved to {ASSETS_DIR}/")
