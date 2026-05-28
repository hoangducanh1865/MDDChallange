import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor

SAMPLE_RATE = 16000
BLANK_TOKEN_ID = 0
PAD_TOKEN_ID = 0


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

def build_vocab(data_dir: str) -> Dict[str, int]:
    csv_path = Path(data_dir) / "metadata" / "train_phones.csv"
    df = pd.read_csv(csv_path)
    tokens = set()
    for col in ("canonical", "transcript"):
        for row in df[col].dropna():
            for tok in str(row).split():
                if tok and tok != "$":
                    tokens.add(tok)
    tokens.add("<eps>")
    # Sort for reproducibility; reserve 0 for blank/pad
    vocab = {"<blank>": 0}
    for i, tok in enumerate(sorted(tokens), start=1):
        vocab[tok] = i
    out_path = Path("outputs") / "vocab.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Built vocab with {len(vocab)} tokens → {out_path}")
    return vocab


def load_or_build_vocab(data_dir: str) -> Dict[str, int]:
    vocab_path = Path("outputs") / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return build_vocab(data_dir)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

def build_feature_extractor() -> Wav2Vec2FeatureExtractor:
    return Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        padding_side="right",
        do_normalize=True,
        return_attention_mask=False,
    )


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def text_to_tensor(string_text: str, vocab: Dict[str, int]) -> List[int]:
    ids = []
    for tok in str(string_text).split():
        if tok and tok != "$":
            if tok in vocab:
                ids.append(vocab[tok])
            # unknown tokens are silently skipped (shouldn't happen at train time)
    return ids


def greedy_decode(logits: torch.Tensor, id2token: Dict[int, str]) -> str:
    pred_ids = torch.argmax(logits, dim=-1).tolist()
    collapsed: List[int] = []
    prev = None
    for tid in pred_ids:
        if tid != prev:
            collapsed.append(tid)
        prev = tid
    tokens: List[str] = []
    for tid in collapsed:
        if tid == BLANK_TOKEN_ID:
            continue
        tok = id2token.get(tid, "")
        if tok and tok not in ("$", "<blank>", "<eps>"):
            tokens.append(tok)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Score computation (writes temp files, calls evaluate.py functions)
# ---------------------------------------------------------------------------

def compute_score(
    gt_df: pd.DataFrame,
    predictions: List[str],
    tmp_dir: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    from src.evaluation.evaluate import compute_f1, compute_per, compute_der

    def _write(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    context = tempfile.TemporaryDirectory() if tmp_dir is None else None
    work_dir = context.name if context else tmp_dir

    gt_rows = [
        {"canonical": str(r["canonical"]), "transcript": str(r["transcript"])}
        for _, r in gt_df.iterrows()
    ]
    pred_rows = [{"predict": p} for p in predictions]

    gt_path   = os.path.join(work_dir, "gt.csv")
    pred_path = os.path.join(work_dir, "pred.csv")
    _write(gt_path, gt_rows)
    _write(pred_path, pred_rows)

    f1  = compute_f1(gt_path, pred_path)
    per = compute_per(gt_path, pred_path)
    der = compute_der(gt_path, pred_path)
    score = 0.5 * f1 + 0.4 * (1.0 - der) + 0.1 * (1.0 - per)

    if context:
        context.cleanup()
    return score, f1, per, der


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_fold_history(
    fold_idx: int,
    model_name: str,
    history: dict,
    out_dir: str = "experiment",
):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [("score", "Score"), ("f1", "F1"), ("per", "PER"), ("der", "DER")]
    for ax, (key, label) in zip(axes.flat, metrics):
        vals = history.get(key, [])
        if vals:
            ax.plot(range(1, len(vals) + 1), vals, marker="o", markersize=3)
        ax.set_title(f"{label} — fold {fold_idx}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    train_loss = history.get("train_loss", [])
    if train_loss:
        axes[0][0].twinx().plot(
            range(1, len(train_loss) + 1), train_loss,
            color="orange", alpha=0.5, linestyle="--", label="train loss",
        )
    fig.suptitle(f"{model_name} — fold {fold_idx}", fontsize=13)
    plt.tight_layout()
    safe_name = model_name.replace("/", "_").replace("-", "_")
    path = os.path.join(out_dir, f"{safe_name}_fold{fold_idx}_metrics.png")
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  Saved plot → {path}")


def visualize_cv_summary(all_results: dict, out_dir: str = "experiment"):
    os.makedirs(out_dir, exist_ok=True)
    model_names = sorted({k[0] for k in all_results})
    n_folds = max(k[1] for k in all_results) + 1

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 3), 5))
    x = np.arange(len(model_names))
    width = 0.6

    for i, mname in enumerate(model_names):
        scores = [all_results.get((mname, f), {}).get("best_score", 0.0) for f in range(n_folds)]
        mean_s = float(np.mean(scores))
        std_s  = float(np.std(scores))
        ax.bar(x[i], mean_s, width, yerr=std_s, capsize=5, label=mname, alpha=0.8)
        ax.text(x[i], mean_s + std_s + 0.005, f"{mean_s:.3f}±{std_s:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in model_names], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Validation Score Summary")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "cv_summary.png")
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  Saved CV summary → {path}")
