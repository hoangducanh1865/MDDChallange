import os
import re
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
    get_device,
    greedy_decode,
    load_or_build_vocab,
    visualize_cv_summary,
    visualize_fold_history,
)


def _safe_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _find_latest_epoch_ckpt(ckpt_dir: Path, fold_idx: int) -> Optional[Path]:
    """Return the highest-numbered fold{i}_epoch{j}.pt in ckpt_dir, or None."""
    candidates = list(ckpt_dir.glob(f"fold{fold_idx}_epoch*.pt"))
    if not candidates:
        return None
    def _epoch_num(p: Path) -> int:
        m = re.search(r"epoch(\d+)", p.name)
        return int(m.group(1)) if m else 0
    return max(candidates, key=_epoch_num)


# ---------------------------------------------------------------------------
# Focal CTC Loss
# ---------------------------------------------------------------------------

class FocalCTCLoss(nn.Module):
    def __init__(self, blank: int = 0, gamma: float = 2.0):
        super().__init__()
        self.blank = blank
        self.gamma = gamma
        self._ctc  = nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)

    def forward(self, logits, targets, input_lengths, target_lengths):
        log_probs  = logits.transpose(0, 1)          # (T, B, V)
        per_sample = self._ctc(log_probs, targets, input_lengths, target_lengths)
        pt         = torch.exp(-per_sample.detach().clamp(max=20))
        weight     = (1.0 - pt) ** self.gamma
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
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = []

    fe_lr = lr_base * (decay ** 12)
    for name, param in model.backbone.feature_extractor.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({"params": [param], "lr": fe_lr, "weight_decay": wd})

    encoder = getattr(model.backbone, "encoder", None)
    if encoder is not None:
        layers = getattr(encoder, "layers", None) or getattr(encoder, "layer", None)
        if layers is not None:
            n = len(layers)
            for layer_idx, layer in enumerate(layers):
                layer_lr = lr_base * (decay ** (n - layer_idx))
                for name, param in layer.named_parameters():
                    if not param.requires_grad:
                        continue
                    wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
                    param_groups.append({"params": [param], "lr": layer_lr, "weight_decay": wd})

    for name, param in model.backbone.named_parameters():
        if any(param is p for g in param_groups for p in g["params"]):
            continue
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({"params": [param], "lr": lr_base, "weight_decay": wd})

    head_lr = lr_base * head_lr_mult
    for module in [model.phonetic_enc, model.linguistic_enc,
                   model.multihead_attn, model.ctc_head, model.binary_head]:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
            param_groups.append({"params": [param], "lr": head_lr, "weight_decay": wd})

    return torch.optim.AdamW(param_groups)


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, optimizer, ctc_loss_fn, binary_loss_fn, device,
                     w_binary=0.1):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        input_values, linguistic, transcripts, target_lengths, wav_lengths, error_labels = batch
        ctc_logits, binary_logits = model(input_values, linguistic)

        log_probs     = F.log_softmax(ctc_logits, dim=2)
        input_lengths = model.get_output_lengths(wav_lengths).clamp(max=log_probs.shape[1])

        loss_ctc = ctc_loss_fn(log_probs, transcripts, input_lengths, target_lengths)
        loss_bin = binary_loss_fn(binary_logits, error_labels)
        loss     = loss_ctc + w_binary * loss_bin

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def _eval_one_epoch(model, loader, val_df, vocab, device):
    model.eval()
    id2token = {v: k for k, v in vocab.items()}
    predictions: List[str] = []

    for batch in tqdm(loader, desc="  eval", leave=False):
        input_values, linguistic, _, _, wav_lengths, _ = batch
        ctc_logits, _ = model(input_values, linguistic)
        log_probs     = F.log_softmax(ctc_logits, dim=2)
        input_lengths = model.get_output_lengths(wav_lengths).clamp(max=log_probs.shape[1])
        for b in range(log_probs.shape[0]):
            hyp = greedy_decode(log_probs[b, :input_lengths[b].item(), :], id2token)
            predictions.append(hyp)

    score, f1, per, der = compute_score(val_df, predictions)
    return score, f1, per, der, predictions


# ---------------------------------------------------------------------------
# Train one fold
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

    safe_name = _safe_name(model_name)
    ckpt_dir  = Path(args.checkpoint_dir) / safe_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Persist model name so inference can recover it without guessing
    (ckpt_dir / "model_name.txt").write_text(model_name)

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx} | Model: {model_name}")
    print(f"  train={len(train_df)}, val={len(val_df)}")
    print(f"  Checkpoints → {ckpt_dir}")

    vocab_size = len(vocab)
    model      = create_model(model_name, vocab_size, device)

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

    ctc_loss_fn    = FocalCTCLoss(blank=BLANK_TOKEN_ID, gamma=args.focal_gamma)
    binary_loss_fn = nn.CrossEntropyLoss()
    optimizer      = build_llrd_optimizer(model, lr_base=args.lr, decay=args.llrd_decay)

    start_epoch = 1
    best_score  = -1.0
    history: Dict[str, list] = {"train_loss": [], "score": [], "f1": [], "per": [], "der": []}

    # ---- Resume from latest epoch checkpoint ----
    resume_ckpt = _find_latest_epoch_ckpt(ckpt_dir, fold_idx)
    if resume_ckpt is not None:
        print(f"  Resuming from {resume_ckpt.name} ...")
        state      = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_score  = state.get("best_score", -1.0)
        history     = state.get("history", history)
        print(f"  Resumed — completed epoch {state['epoch']}, best_score={best_score:.4f}")

    if start_epoch > args.epochs:
        print(f"  Fold {fold_idx} already completed ({args.epochs} epochs), skipping.")
        best_ckpt_path = str(ckpt_dir / f"fold{fold_idx}_best.pt")
        return best_score, history, best_ckpt_path

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, ctc_loss_fn, binary_loss_fn, device,
        )
        history["train_loss"].append(train_loss)

        if epoch >= 3:
            score, f1, per, der, _ = _eval_one_epoch(
                model, val_loader, val_df, vocab, device,
            )
            history["score"].append(score)
            history["f1"].append(f1)
            history["per"].append(per)
            history["der"].append(der)

            print(
                f"  Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | "
                f"Score={score:.4f}  F1={f1:.4f}  DER={der:.4f}  PER={per:.4f}"
            )

            # Always update fold{i}_best.pt when score improves
            if score > best_score:
                best_score = score
                torch.save(
                    model.state_dict(),
                    ckpt_dir / f"fold{fold_idx}_best.pt",
                )
                print(f"  ✓ Updated best (Score={best_score:.4f})")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f}")

        # Save full state every epoch for resuming
        torch.save(
            {
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "epoch":      epoch,
                "best_score": best_score,
                "history":    history,
            },
            ckpt_dir / f"fold{fold_idx}_epoch{epoch}.pt",
        )

    visualize_fold_history(fold_idx, model_name, history)
    return best_score, history, str(ckpt_dir / f"fold{fold_idx}_best.pt")


# ---------------------------------------------------------------------------
# Cross-Validation entry point
# ---------------------------------------------------------------------------

def run_cross_validation(args):
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    vocab             = load_or_build_vocab(args.data_dir)
    feature_extractor = build_feature_extractor()
    splits            = get_kfold_splits(args.data_dir, args.n_folds, args.seed)

    model_name   = args.model
    all_results  = {}
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
        torch.cuda.empty_cache()

    mean_s = float(np.mean(model_scores))
    std_s  = float(np.std(model_scores))

    print("\n" + "=" * 60)
    print(f"[{model_name}]  Score: {mean_s:.4f} ± {std_s:.4f}")
    print("=" * 60)

    visualize_cv_summary(all_results)
    return all_results
