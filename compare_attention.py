"""
Attention map comparison: Base LLaMA vs Register+LoRA.

Generates side-by-side attention visualizations showing how register tokens
redistribute attention away from BOS.

Run:
  python compare_attention.py
"""

import os, sys, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

sys.path.insert(0, "/home/gyeongtae/selfcheckgpt")

from transformers import LlamaForCausalLM, AutoTokenizer
from register_llama.config import Config
from register_llama.model import load_tokenizer, get_register_ids
from register_llama.dataset import _insert_registers
from register_llama.evaluate import load_trained_model

CHECKPOINT = "/home/gyeongtae/selfcheckgpt/checkpoints/final"
OUT_DIR = "/home/gyeongtae/selfcheckgpt/results/images/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_TEXTS = [
    "Terry's daughter was in extreme pain in her mouth. After an evaluation, "
    "Terry realized that her daughter was teething. Once Terry realized she was "
    "teething, Terry administered medication.",

    "Bill was not the most confident driver. It was time for him to take his "
    "road test for his license. He had a little trouble backing out of the "
    "parking space. After that, it went great.",
]

MAX_LEN = 60   # truncate tokens for readable plots


# ── Attention extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def get_attn_base(model, tokenizer, text, device, max_len):
    ids = tokenizer.encode(text, add_special_tokens=True)[:max_len]
    tokens = ["[BOS]"] + [tokenizer.decode([i])[:6] for i in ids[1:]]
    input_ids = torch.tensor([ids], device=device)
    out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attn = np.stack([a[0].float().cpu().numpy() for a in out.attentions])  # (L,H,S,S)
    return attn, tokens


@torch.no_grad()
def get_attn_register(model, tokenizer, register_ids, text, device, max_len):
    raw = tokenizer.encode(text, add_special_tokens=True)
    ids = _insert_registers(raw, tokenizer.bos_token_id, register_ids)[:max_len]
    n_reg = len(register_ids)
    tokens = ["[BOS]"] + [f"[R{i+1}]" for i in range(n_reg)]
    for tid in ids[1 + n_reg:]:
        tokens.append(tokenizer.decode([tid])[:6])
    input_ids = torch.tensor([ids], device=device)
    inner = model.base_model.model if hasattr(model, "base_model") else model
    out = inner(input_ids=input_ids, output_attentions=True, use_cache=False)
    attn = np.stack([a[0].float().cpu().numpy() for a in out.attentions])  # (L,H,S,S)
    return attn, tokens, n_reg


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _label_axes(ax, xtokens, ytokens, title, fontsize=6):
    ax.set_xticks(range(len(xtokens)))
    ax.set_xticklabels(xtokens, rotation=90, fontsize=fontsize)
    ax.set_yticks(range(len(ytokens)))
    ax.set_yticklabels(ytokens, fontsize=fontsize)
    ax.set_title(title, fontsize=9, pad=4)


# ── Figure 1: Side-by-side full attention matrix ───────────────────────────────

def fig_attention_matrix(attn_base, tokens_base, attn_reg, tokens_reg, n_reg,
                          layer=15, head=0, save_path=None):
    A_base = attn_base[layer, head]
    A_reg  = attn_reg[layer, head]

    vmax = max(A_base.max(), A_reg.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Full Attention Matrix — Layer {layer}, Head {head}", fontsize=13)

    # Base LLaMA
    im0 = axes[0].imshow(A_base, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    _label_axes(axes[0], tokens_base, tokens_base, "Base LLaMA (no register)")
    plt.colorbar(im0, ax=axes[0], shrink=0.6)

    # Register+LoRA
    im1 = axes[1].imshow(A_reg, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    _label_axes(axes[1], tokens_reg, tokens_reg, "Register + LoRA")
    plt.colorbar(im1, ax=axes[1], shrink=0.6)

    # Diff: attention to BOS column (col 0) across all query positions
    bos_base = A_base[:, 0]   # (S_base,)
    bos_reg  = A_reg[:, 0]    # (S_reg,)  — BOS is still col 0

    ax2 = axes[2]
    x_base = np.arange(len(bos_base))
    x_reg  = np.arange(len(bos_reg))
    ax2.plot(x_base, bos_base, label="Base LLaMA → BOS", color="#4C72B0", alpha=0.8)
    ax2.plot(x_reg,  bos_reg,  label="Register+LoRA → BOS", color="#DD8452", alpha=0.8)
    ax2.set_xlabel("Query token position", fontsize=10)
    ax2.set_ylabel("Attention weight to BOS", fontsize=10)
    ax2.set_title("BOS Attention per Query Position", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ── Figure 2: Per-layer BOS & sink attention ───────────────────────────────────

def fig_per_layer_sink(attn_base, attn_reg, n_reg, save_path=None):
    """Compare attention received by BOS (and registers) across all 32 layers."""
    n_layers = attn_base.shape[0]

    # Mean over heads, then mean over query positions (col-wise = "attention received")
    col_base = attn_base.mean(axis=1).mean(axis=1)   # (L, S_base)
    col_reg  = attn_reg.mean(axis=1).mean(axis=1)    # (L, S_reg)

    bos_base    = col_base[:, 0]
    bos_reg     = col_reg[:, 0]
    reg_mean    = col_reg[:, 1:1+n_reg].mean(axis=1)  # mean of REG1-4
    content_base = col_base[:, 1:].mean(axis=1)
    content_reg  = col_reg[:, 1+n_reg:].mean(axis=1)

    layers = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Attention Distribution per Layer: Base LLaMA vs Register+LoRA", fontsize=13)

    # Left: BOS attention comparison
    axes[0].plot(layers, bos_base, label="Base — BOS",        color="#4C72B0", linewidth=2)
    axes[0].plot(layers, bos_reg,  label="Register — BOS",    color="#4C72B0", linewidth=2,
                 linestyle="--", alpha=0.7)
    axes[0].plot(layers, reg_mean, label="Register — REG avg", color="#DD8452", linewidth=2)
    axes[0].set_xlabel("Layer", fontsize=11)
    axes[0].set_ylabel("Mean attention received", fontsize=11)
    axes[0].set_title("BOS vs Register Attention per Layer", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Right: content token attention
    axes[1].plot(layers, content_base, label="Base — Content",     color="#55A868", linewidth=2)
    axes[1].plot(layers, content_reg,  label="Register — Content", color="#55A868", linewidth=2,
                 linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Layer", fontsize=11)
    axes[1].set_ylabel("Mean attention received", fontsize=11)
    axes[1].set_title("Content Token Attention per Layer", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ── Figure 3: Sink distribution stacked bar ────────────────────────────────────

def fig_sink_distribution(attn_base, attn_reg, n_reg, save_path=None):
    """Show how attention mass is split: BOS / REG / Content for each model."""
    col_base = attn_base.mean(axis=(0, 1, 2))   # mean over L, H, query → (S_base,)
    col_reg  = attn_reg.mean(axis=(0, 1, 2))    # (S_reg,)

    # Base: BOS + content
    base_bos     = col_base[0]
    base_content = col_base[1:].mean()

    # Register: BOS + REG1~4 + content
    reg_bos     = col_reg[0]
    reg_regs    = col_reg[1:1+n_reg]
    reg_content = col_reg[1+n_reg:].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Attention Sink Distribution: Base LLaMA vs Register+LoRA", fontsize=13)

    # Left: stacked bar
    models   = ["Base LLaMA", "Register+LoRA"]
    bos_vals = [base_bos, reg_bos]
    reg_vals = [0, reg_regs.mean()]
    con_vals = [base_content, reg_content]

    x = np.arange(2)
    w = 0.5
    p1 = axes[0].bar(x, bos_vals, w, label="BOS",          color="#4C72B0")
    p2 = axes[0].bar(x, reg_vals, w, bottom=bos_vals,       label="REG (avg)", color="#DD8452")
    p3 = axes[0].bar(x, con_vals, w,
                     bottom=[b+r for b, r in zip(bos_vals, reg_vals)],
                     label="Content", color="#55A868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=12)
    axes[0].set_ylabel("Mean attention weight", fontsize=11)
    axes[0].set_title("Aggregate Attention Distribution", fontsize=11)
    axes[0].legend(fontsize=10)
    for bar, val in zip(p1, bos_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, val/2,
                     f"{val:.4f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # Right: per-register breakdown
    reg_labels = [f"REG{i+1}" for i in range(n_reg)]
    reg_colors = ["#DD8452", "#C44E52", "#8172B2", "#937860"]
    axes[1].bar(["BOS\n(Base)"] + reg_labels + ["BOS\n(Reg)"],
                [base_bos] + list(reg_regs) + [reg_bos],
                color=["#4C72B0"] + reg_colors + ["#4C72B0"],
                edgecolor="white")
    axes[1].set_ylabel("Mean attention received", fontsize=11)
    axes[1].set_title("BOS vs Each Register Token", fontsize=11)
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate([base_bos] + list(reg_regs) + [reg_bos]):
        axes[1].text(i, v + 0.0005, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ── Figure 4: Head-averaged heatmap grid (select layers) ──────────────────────

def fig_layer_grid(attn_base, tokens_base, attn_reg, tokens_reg,
                   layers=(0, 4, 8, 15, 20, 28, 31), save_path=None):
    """Show head-averaged attention matrix at key layers, side by side."""
    n = len(layers)
    fig, axes = plt.subplots(2, n, figsize=(3.5*n, 7))
    fig.suptitle("Head-Averaged Attention at Key Layers\n(Top: Base LLaMA  |  Bottom: Register+LoRA)",
                 fontsize=12)

    vmax = max(
        attn_base.mean(axis=1).max(),
        attn_reg.mean(axis=1).max(),
    )

    for col, l in enumerate(layers):
        A_base = attn_base[l].mean(axis=0)
        A_reg  = attn_reg[l].mean(axis=0)

        axes[0, col].imshow(A_base, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        axes[0, col].set_title(f"Layer {l}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(A_reg, cmap="Oranges", vmin=0, vmax=vmax, aspect="auto")
        axes[1, col].axis("off")

    axes[0, 0].axis("on")
    axes[0, 0].set_yticks([])
    axes[0, 0].set_xticks([])
    axes[0, 0].set_ylabel("Base LLaMA", fontsize=10)

    axes[1, 0].axis("on")
    axes[1, 0].set_yticks([])
    axes[1, 0].set_xticks([])
    axes[1, 0].set_ylabel("Register+LoRA", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ── Figure 5: Attention difference (BOS column) heatmap ───────────────────────

def fig_bos_diff_heatmap(attn_base, attn_reg, save_path=None):
    """
    For each layer × head, show: BOS attention in Base minus BOS in Register.
    Positive = base had more BOS attention → registers redistributed it.
    """
    # BOS column: mean attention TO position 0 across all query positions
    # attn shape: (L, H, S, S); [:, :, :, 0].mean(axis=2) → (L, H)
    bos_base = attn_base[:, :, :, 0].mean(axis=2)   # (L, H)
    bos_reg  = attn_reg[:, :, :, 0].mean(axis=2)    # (L, H)
    diff = bos_base - bos_reg                         # positive = more BOS in base

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("BOS Attention: Base LLaMA vs Register+LoRA (per Layer × Head)", fontsize=12)

    vmax = max(abs(bos_base).max(), abs(bos_reg).max())
    vdiff = abs(diff).max()

    im0 = axes[0].imshow(bos_base.T, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    axes[0].set_title("Base LLaMA — BOS attention", fontsize=10)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Head")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].imshow(bos_reg.T, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    axes[1].set_title("Register+LoRA — BOS attention", fontsize=10)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Head")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)

    im2 = axes[2].imshow(diff.T, cmap="RdBu_r", vmin=-vdiff, vmax=vdiff, aspect="auto")
    axes[2].set_title("Difference (Base − Register)\nRed = registers absorbed more", fontsize=10)
    axes[2].set_xlabel("Layer"); axes[2].set_ylabel("Head")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    device = torch.device("cuda:0")
    text = SAMPLE_TEXTS[0]
    print(f"Sample: {text[:80]}…\n")

    # ── Load base LLaMA, extract, then free VRAM ──────────────────────────────
    print("Loading Base LLaMA…")
    base_tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = LlamaForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    base_model.eval()
    attn_base, tokens_base = get_attn_base(base_model, base_tokenizer, text, device, MAX_LEN)
    print(f"  Base attention shape: {attn_base.shape}")
    del base_model
    torch.cuda.empty_cache()

    # ── Load Register+LoRA, extract, then free VRAM ───────────────────────────
    print("Loading Register+LoRA…")
    reg_tokenizer = load_tokenizer(cfg)
    register_ids = get_register_ids(reg_tokenizer, cfg)
    reg_model = load_trained_model(CHECKPOINT, cfg, reg_tokenizer, attn_implementation="eager")
    reg_model.eval()
    attn_reg, tokens_reg, n_reg = get_attn_register(reg_model, reg_tokenizer, register_ids,
                                                      text, device, MAX_LEN)
    print(f"  Register attention shape: {attn_reg.shape}")
    del reg_model
    torch.cuda.empty_cache()

    print("[2/5] Figure 1: Side-by-side attention matrix…")
    fig_attention_matrix(attn_base, tokens_base, attn_reg, tokens_reg, n_reg,
                         layer=15, head=0,
                         save_path=f"{OUT_DIR}/fig1_matrix_layer15.png")

    print("[3/5] Figure 2: Per-layer sink attention…")
    fig_per_layer_sink(attn_base, attn_reg, n_reg,
                       save_path=f"{OUT_DIR}/fig2_per_layer_sink.png")

    print("[4/5] Figure 3: Sink distribution…")
    fig_sink_distribution(attn_base, attn_reg, n_reg,
                          save_path=f"{OUT_DIR}/fig3_sink_distribution.png")

    print("[5/5] Figure 4: Layer grid + Figure 5: BOS diff heatmap…")
    fig_layer_grid(attn_base, tokens_base, attn_reg, tokens_reg,
                   layers=(0, 4, 8, 15, 20, 28, 31),
                   save_path=f"{OUT_DIR}/fig4_layer_grid.png")
    fig_bos_diff_heatmap(attn_base, attn_reg,
                         save_path=f"{OUT_DIR}/fig5_bos_diff.png")

    print(f"\nAll figures saved to {OUT_DIR}/")
