import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MDDDataset, get_kfold_splits, make_collate_fn
from src.model import MDDModel, create_model
from src.utils import (
    BLANK_TOKEN_ID,
    build_feature_extractor,
    compute_score,
    greedy_decode,
    load_or_build_vocab,
    visualize_cv_summary,
    visualize_fold_history,
)

BACKBONE_NAMES = [
    "wav2vec2-base-100h",
    "wav2vec2-base-vietnamese-250h",
    "hubert-base-ls960",
]


# ---------------------------------------------------------------------------
# Focal CTC Loss
# ---------------------------------------------------------------------------

class FocalCTCLoss(nn.Module):
    """CTC loss with per-sample focal reweighting to emphasise hard examples."""

    def __init__(self, blank: int = 0, gamma: float = 2.0):
        super().__init__()
        self.blank = blank
        self.gamma = gamma
        self._ctc  = nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)

    def forward(
        self,
        logits: torch.Tensor,       # (B, T, V) — log_softmax already applied
        targets: torch.Tensor,       # (B, max_len)
        input_lengths: torch.Tensor, # (B,)
        target_lengths: torch.Tensor,# (B,)
    ) -> torch.Tensor:
        # CTCLoss expects (T, B, V)
        log_probs = logits.transpose(0, 1)  # (T, B, V)
        per_sample = self._ctc(log_probs, targets, input_lengths, target_lengths)
        # Focal weight: (1 - exp(-loss))^gamma  (larger loss → higher weight)
        pt = torch.exp(-per_sample.detach().clamp(max=20))
        weight = (1.0 - pt) ** self.gamma
        return (weight * per_sample).mean()


# ---------------------------------------------------------------------------
# LLRD Optimizer
# ---------------------------------------------------------------------------

def build_llrd_optimizer(
    model: MDDModel,
    lr_base: float,
    decay: float = 0.9,
    head_lr_mult: float = 10.0,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Layer-wise learning rate decay for the transformer backbone."""
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = []

    # --- feature extractor (CNN) — lowest LR
    fe_lr = lr_base * (decay ** 12)
    for name, param in model.backbone.feature_extractor.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({"params": [param], "lr": fe_lr, "weight_decay": wd})

    # --- transformer encoder layers
    encoder = getattr(model.backbone, "encoder", None)
    if encoder is not None:
        layers = getattr(encoder, "layers", None)
        if layers is None:
            layers = getattr(encoder, "layer", None)
        if layers is not None:
            n = len(layers)
            for layer_idx, layer in enumerate(layers):
                layer_lr = lr_base * (decay ** (n - layer_idx))
                for name, param in layer.named_parameters():
                    if not param.requires_grad:
                        continue
                    wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
                    param_groups.append({"params": [param], "lr": layer_lr, "weight_decay": wd})

    # --- projection / pos-encoding layers of backbone
    backbone_extra_names = {
        "feature_projection", "masked_spec_embed",
        "pos_conv_embed", "layer_norm", "encoder.layer_norm",
    }
    for name, param in model.backbone.named_parameters():
        already_added = any(param is p for g in param_groups for p in g["params"])
        if already_added or not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({"params": [param], "lr": lr_base, "weight_decay": wd})

    # --- heads and aux encoders — highest LR
    head_lr = lr_base * head_lr_mult
    head_modules = [
        model.phonetic_enc, model.linguistic_enc,
        model.multihead_attn, model.ctc_head, model.binary_head,
    ]
    for module in head_modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
            param_groups.append({"params": [param], "lr": head_lr, "weight_decay": wd})

    return torch.optim.AdamW(param_groups)


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: MDDModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc_loss_fn: FocalCTCLoss,
    binary_loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
    w_binary: float = 0.1,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        input_values, linguistic, transcripts, target_lengths, wav_lengths, error_labels = batch

        ctc_logits, binary_logits = model(input_values, linguistic)

        # CTC
        log_probs    = F.log_softmax(ctc_logits, dim=2)
        input_lengths= model.get_output_lengths(wav_lengths)
        input_lengths= input_lengths.clamp(max=log_probs.shape[1])

        loss_ctc  = ctc_loss_fn(log_probs, transcripts, input_lengths, target_lengths)
        loss_bin  = binary_loss_fn(binary_logits, error_labels)
        loss      = loss_ctc + w_binary * loss_bin

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def _eval_one_epoch(
    model: MDDModel,
    loader: DataLoader,
    val_df: pd.DataFrame,
    vocab: Dict[str, int],
    device: torch.device,
) -> Tuple[float, float, float, float, List[str]]:
    model.eval()
    id2token = {v: k for k, v in vocab.items()}
    predictions: List[str] = []

    for batch in tqdm(loader, desc="  eval", leave=False):
        input_values, linguistic, transcripts, target_lengths, wav_lengths, _ = batch
        ctc_logits, _ = model(input_values, linguistic)
        log_probs = F.log_softmax(ctc_logits, dim=2)
        input_lengths = model.get_output_lengths(wav_lengths).clamp(max=log_probs.shape[1])

        for b in range(log_probs.shape[0]):
            valid_len = input_lengths[b].item()
            hyp = greedy_decode(log_probs[b, :valid_len, :], id2token)
            predictions.append(hyp)

    score, f1, per, der = compute_score(val_df, predictions)
    return score, f1, per, der, predictions


# ---------------------------------------------------------------------------
# Train one fold for one backbone
# ---------------------------------------------------------------------------

def train_fold(
    fold_idx: int,
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vocab: Dict[str, int],
    feature_extractor,
    args,
    device: torch.device,
) -> Tuple[float, dict, str]:
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx} | Model: {model_name}")
    print(f"  train={len(train_df)}, val={len(val_df)}")

    vocab_size = len(vocab)
    model = create_model(model_name, vocab_size, device)

    # Data
    train_ds = MDDDataset(train_df, args.data_dir, vocab, augment=True)
    val_ds   = MDDDataset(val_df,   args.data_dir, vocab, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate_fn(feature_extractor, device, spec_augment=True),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
        collate_fn=make_collate_fn(feature_extractor, device, spec_augment=False),
        num_workers=0,
    )

    # Loss & optimizer
    ctc_loss_fn    = FocalCTCLoss(blank=BLANK_TOKEN_ID, gamma=args.focal_gamma)
    binary_loss_fn = nn.CrossEntropyLoss()
    optimizer      = build_llrd_optimizer(model, lr_base=args.lr, decay=args.llrd_decay)

    best_score = -1.0
    best_ckpt  = ""
    history    = {"train_loss": [], "score": [], "f1": [], "per": [], "der": []}

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace("-", "_")

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, ctc_loss_fn, binary_loss_fn, device,
        )
        history["train_loss"].append(train_loss)

        # Always evaluate (skip only first 2 epochs to save time)
        if epoch >= 3:
            score, f1, per, der, _ = _eval_one_epoch(model, val_loader, val_df, vocab, device)
            history["score"].append(score)
            history["f1"].append(f1)
            history["per"].append(per)
            history["der"].append(der)

            print(
                f"  Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | "
                f"Score={score:.4f}  F1={f1:.4f}  DER={der:.4f}  PER={per:.4f}"
            )

            if score > best_score:
                best_score = score
                best_ckpt  = str(ckpt_dir / f"best_fold{fold_idx}_{safe_name}.pt")
                torch.save(model.state_dict(), best_ckpt)
                print(f"  ✓ Saved checkpoint (Score={best_score:.4f}) → {best_ckpt}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f}")

    visualize_fold_history(fold_idx, model_name, history)
    return best_score, history, best_ckpt


# ---------------------------------------------------------------------------
# 5-Fold Cross Validation (all backbones)
# ---------------------------------------------------------------------------

def run_cross_validation(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab            = load_or_build_vocab(args.data_dir)
    feature_extractor= build_feature_extractor()
    splits           = get_kfold_splits(args.data_dir, args.n_folds, args.seed)

    all_results: dict = {}

    for model_name in BACKBONE_NAMES:
        model_scores = []
        for fold_idx, (train_df, val_df) in enumerate(splits):
            best_score, history, ckpt = train_fold(
                fold_idx, model_name, train_df, val_df,
                vocab, feature_extractor, args, device,
            )
            all_results[(model_name, fold_idx)] = {
                "best_score": best_score,
                "history":    history,
                "checkpoint": ckpt,
            }
            model_scores.append(best_score)
            # Free GPU memory between folds
            torch.cuda.empty_cache()

        mean_s = float(np.mean(model_scores))
        std_s  = float(np.std(model_scores))
        print(f"\n[{model_name}] Score: {mean_s:.4f} ± {std_s:.4f}")

    # Final summary
    print("\n" + "="*70)
    print("Cross-Validation Summary")
    print("="*70)
    for model_name in BACKBONE_NAMES:
        scores = [all_results[(model_name, f)]["best_score"] for f in range(args.n_folds)]
        print(f"  {model_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    visualize_cv_summary(all_results)
    return all_results
