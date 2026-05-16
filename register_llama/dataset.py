"""
Datasets for ROCStories (train), Story Cloze Test (val/test),
and CNN/DailyMail (train + SelfCheckGPT eval).
"""

import os
import glob
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple

from register_llama.config import Config


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_csvs(directory: str) -> List[pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return [pd.read_csv(p) for p in paths]


def _insert_registers(token_ids: List[int], bos_id: int, register_ids: List[int]) -> List[int]:
    """
    [BOS, t1, t2, ...] → [BOS, REG1, REG2, REG3, REG4, t1, t2, ...]
    Assumes the first element is already the BOS token.
    """
    if token_ids and token_ids[0] == bos_id:
        return [token_ids[0]] + register_ids + token_ids[1:]
    return [bos_id] + register_ids + token_ids


def _tokenize_with_registers(
    text: str,
    tokenizer,
    register_ids: List[int],
    max_length: int,
) -> List[int]:
    raw = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    ids = _insert_registers(raw, tokenizer.bos_token_id, register_ids)
    return ids[:max_length]


# ── ROCStories (CLM training) ──────────────────────────────────────────────

class ROCStoriesDataset(Dataset):
    """
    5-sentence stories → causal LM targets.
    Register tokens are inserted after BOS; their positions are masked in labels.
    """

    def __init__(self, cfg: Config, tokenizer, register_ids: List[int]):
        self.tokenizer = tokenizer
        self.register_ids = register_ids
        self.max_length = cfg.max_length
        self.pad_id = tokenizer.pad_token_id
        self.n_prefix = 1 + len(register_ids)  # BOS + REGs

        frames = _load_csvs(cfg.train_dir)
        self.stories: List[str] = []
        for df in frames:
            for _, row in df.iterrows():
                sents = [str(row[f"sentence{i}"]) for i in range(1, 6)]
                self.stories.append(" ".join(sents))

        print(f"ROCStoriesDataset: {len(self.stories):,} stories loaded")

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        ids = _tokenize_with_registers(
            self.stories[idx], self.tokenizer, self.register_ids, self.max_length
        )
        seq_len = len(ids)
        pad_len = self.max_length - seq_len

        # Labels: -100 for BOS+REGs (no loss on register positions)
        labels = [-100] * self.n_prefix + ids[self.n_prefix:] + [-100] * pad_len

        attention_mask = [1] * seq_len + [0] * pad_len
        ids = ids + [self.pad_id] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ── Story Cloze Test (val / test) ─────────────────────────────────────────

class ClozeTestDataset(Dataset):
    """
    Each item: story context (4 sentences) + two candidate endings.
    AnswerRightEnding (1 or 2) is the label when available.
    """

    def __init__(self, directory: str, tokenizer, register_ids: List[int], max_length: int = 256):
        self.tokenizer = tokenizer
        self.register_ids = register_ids
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        self.n_prefix = 1 + len(register_ids)

        frames = _load_csvs(directory)
        self.items: List[dict] = []
        for df in frames:
            for _, row in df.iterrows():
                context = " ".join([
                    str(row["InputSentence1"]), str(row["InputSentence2"]),
                    str(row["InputSentence3"]), str(row["InputSentence4"]),
                ])
                label = int(row["AnswerRightEnding"]) if "AnswerRightEnding" in row else -1
                self.items.append({
                    "context": context,
                    "ending1": str(row["RandomFifthSentenceQuiz1"]),
                    "ending2": str(row["RandomFifthSentenceQuiz2"]),
                    "label": label,
                })

        has_labels = sum(1 for x in self.items if x["label"] != -1)
        print(f"ClozeTestDataset ({directory}): {len(self.items):,} examples, "
              f"{has_labels} with labels")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ── ClozeTest evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def compute_nll(
    model,
    tokenizer,
    register_ids: List[int],
    context: str,
    ending: str,
    max_length: int,
    device: torch.device,
) -> float:
    """
    NLL of `ending` conditioned on `context` (lower = model prefers this ending).
    Loss is computed only on the ending tokens, not the context or register tokens.
    """
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    end_ids = tokenizer.encode(ending, add_special_tokens=False)

    bos = tokenizer.bos_token_id
    all_ids = [bos] + register_ids + ctx_ids + end_ids

    # Truncate context if too long, preserve ending
    n_prefix = 1 + len(register_ids)
    n_end = len(end_ids)
    if len(all_ids) > max_length:
        max_ctx = max_length - n_prefix - n_end
        ctx_ids = ctx_ids[-max_ctx:]
        all_ids = [bos] + register_ids + ctx_ids + end_ids

    # Mask everything before the ending tokens
    n_before_end = len(all_ids) - n_end
    labels = [-100] * n_before_end + end_ids

    input_ids = torch.tensor([all_ids], device=device)
    labels_t = torch.tensor([labels], device=device)

    out = model(input_ids=input_ids, labels=labels_t)
    return out.loss.item()


@torch.no_grad()
def evaluate_cloze_accuracy(
    model,
    tokenizer,
    register_ids: List[int],
    dataset: ClozeTestDataset,
    device: torch.device,
    max_length: int = 256,
    verbose: bool = False,
) -> Tuple[float, int]:
    """Returns (accuracy, n_labeled_examples)."""
    model.eval()
    correct = total = 0

    for item in dataset.items:
        if item["label"] == -1:
            continue

        nll1 = compute_nll(model, tokenizer, register_ids, item["context"],
                           item["ending1"], max_length, device)
        nll2 = compute_nll(model, tokenizer, register_ids, item["context"],
                           item["ending2"], max_length, device)

        pred = 1 if nll1 < nll2 else 2
        if pred == item["label"]:
            correct += 1
        total += 1

        if verbose and total <= 3:
            print(f"  NLL1={nll1:.3f} NLL2={nll2:.3f} pred={pred} label={item['label']}")

    acc = correct / total if total > 0 else 0.0
    return acc, total


# ── CNN/DailyMail (CLM training) ──────────────────────────────────────────

class CNNDailyMailDataset(Dataset):
    """
    CNN/DailyMail articles → causal LM targets.
    Register tokens inserted after BOS; register positions masked in labels.
    """

    def __init__(self, split: str, cfg: Config, tokenizer, register_ids: List[int]):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.register_ids = register_ids
        self.max_length = cfg.max_length
        self.pad_id = tokenizer.pad_token_id
        self.n_prefix = 1 + len(register_ids)

        ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
        self.articles: List[str] = [item["article"] for item in ds]

        print(f"CNNDailyMailDataset ({split}): {len(self.articles):,} articles loaded")

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        ids = _tokenize_with_registers(
            self.articles[idx], self.tokenizer, self.register_ids, self.max_length
        )
        seq_len = len(ids)
        pad_len = self.max_length - seq_len

        labels = [-100] * self.n_prefix + ids[self.n_prefix:] + [-100] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
        ids = ids + [self.pad_id] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CNNDailyMailEvalDataset(Dataset):
    """
    For SelfCheckGPT evaluation on CNN/DailyMail.
    Uses the first `prompt_tokens` tokens of each article as the generation prompt.
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        register_ids: List[int],
        n_examples: int = 50,
        prompt_tokens: int = 64,
        seed: int = 42,
    ):
        from datasets import load_dataset

        ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
        random.seed(seed)
        indices = random.sample(range(len(ds)), min(n_examples, len(ds)))

        self.tokenizer = tokenizer
        self.register_ids = register_ids
        self.items: List[dict] = []

        for idx in indices:
            article = ds[int(idx)]["article"]
            raw_tokens = tokenizer.encode(article, add_special_tokens=False)
            prompt_tok = raw_tokens[:prompt_tokens]
            prompt_text = tokenizer.decode(prompt_tok, skip_special_tokens=True)
            self.items.append({
                "prompt": prompt_text,
                "full_article": article,
            })

        print(f"CNNDailyMailEvalDataset ({split}): {len(self.items)} examples "
              f"(prompt_tokens={prompt_tokens})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
