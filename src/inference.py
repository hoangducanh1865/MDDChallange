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
from src.train import BACKBONE_NAMES
from src.utils import (
    BLANK_TOKEN_ID,
    build_feature_extractor,
    compute_score,
    greedy_decode,
    load_or_build_vocab,
)


# ---------------------------------------------------------------------------
# Temperature Scaling calibration
# ---------------------------------------------------------------------------

class _TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)


def temperature_scale(
    model: MDDModel,
    val_loader: DataLoader,
    vocab: Dict[str, int],
    device: torch.device,
    n_steps: int = 50,
    lr: float = 0.05,
) -> float:
    """Find optimal temperature on validation set. Returns scalar T."""
    model.eval()
    scaler = _TemperatureScaler().to(device)

    all_logits:      List[torch.Tensor] = []
    all_targets:     List[torch.Tensor] = []
    all_input_lens:  List[torch.Tensor] = []
    all_target_lens: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            input_values, linguistic, transcripts, target_lengths, wav_lengths, _ = batch
            ctc_logits, _ = model(input_values, linguistic)
            input_lengths = model.get_output_lengths(wav_lengths).clamp(max=ctc_logits.shape[1])
            all_logits.append(ctc_logits.cpu())
            all_targets.append(transcripts.cpu())
            all_input_lens.append(input_lengths.cpu())
            all_target_lens.append(target_lengths.cpu())

    if not all_logits:
        return 1.0

    cat_logits     = torch.cat(all_logits,     dim=0).to(device)
    cat_targets    = torch.cat(all_targets,    dim=0).to(device)
    cat_input_lens = torch.cat(all_input_lens, dim=0).to(device)
    cat_target_lens= torch.cat(all_target_lens,dim=0).to(device)

    ctc_loss_fn = nn.CTCLoss(blank=BLANK_TOKEN_ID, reduction="mean", zero_infinity=True)
    optimizer   = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=n_steps)

    def closure():
        optimizer.zero_grad()
        scaled     = scaler(cat_logits)
        log_probs  = F.log_softmax(scaled, dim=2).transpose(0, 1)
        loss       = ctc_loss_fn(log_probs, cat_targets, cat_input_lens, cat_target_lens)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = float(scaler.temperature.clamp(min=0.05).item())
    print(f"  Calibrated temperature: {T:.4f}")
    return T


# ---------------------------------------------------------------------------
# Greedy inference for a DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: MDDModel,
    loader: DataLoader,
    vocab: Dict[str, int],
    device: torch.device,
    temperature: float = 1.0,
) -> List[str]:
    model.eval()
    id2token = {v: k for k, v in vocab.items()}
    predictions: List[str] = []
    for batch in tqdm(loader, desc="  inference", leave=False):
        input_values, linguistic, _, _, wav_lengths, _ = batch
        ctc_logits, _ = model(input_values, linguistic)
        log_probs = F.log_softmax(ctc_logits / max(temperature, 0.05), dim=2)
        input_lengths = model.get_output_lengths(wav_lengths).clamp(max=log_probs.shape[1])
        for b in range(log_probs.shape[0]):
            hyp = greedy_decode(log_probs[b, :input_lengths[b].item(), :], id2token)
            predictions.append(hyp)
    return predictions


# ---------------------------------------------------------------------------
# Collect log-probability arrays from a model on a DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_logits(
    model: MDDModel,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> List[np.ndarray]:
    """Return list of per-sample log-prob arrays (shape: T_i × V)."""
    model.eval()
    result: List[np.ndarray] = []
    for batch in tqdm(loader, desc="  collect logits", leave=False):
        input_values, linguistic, _, _, wav_lengths, _ = batch
        ctc_logits, _ = model(input_values, linguistic)
        log_probs = F.log_softmax(ctc_logits / max(temperature, 0.05), dim=2)
        input_lengths = model.get_output_lengths(wav_lengths).clamp(max=log_probs.shape[1])
        for b in range(log_probs.shape[0]):
            vl = input_lengths[b].item()
            result.append(log_probs[b, :vl, :].cpu().numpy())
    return result


def _decode_from_logits(
    per_sample_logits: List[np.ndarray],
    id2token: Dict[int, str],
) -> List[str]:
    predictions = []
    for lg in per_sample_logits:
        hyp = greedy_decode(torch.tensor(lg), id2token)
        predictions.append(hyp)
    return predictions


def _avg_logits(fold_arrays: List[np.ndarray]) -> np.ndarray:
    """Average multiple (T, V) arrays, padding shorter ones with zeros."""
    max_t = max(a.shape[0] for a in fold_arrays)
    v     = fold_arrays[0].shape[1]
    stacked = np.zeros((len(fold_arrays), max_t, v), dtype=np.float32)
    for i, a in enumerate(fold_arrays):
        stacked[i, :a.shape[0], :] = a
    return stacked.mean(axis=0)[:max_t, :]


# ---------------------------------------------------------------------------
# Optuna ensemble weight search
# ---------------------------------------------------------------------------

def find_ensemble_weights_optuna(
    model_avg_logits: Dict[str, List[np.ndarray]],
    val_df: pd.DataFrame,
    vocab: Dict[str, int],
    n_trials: int = 100,
) -> Dict[str, float]:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("optuna not installed — using equal weights")
        n = len(model_avg_logits)
        return {m: 1.0 / n for m in model_avg_logits}

    id2token  = {v: k for k, v in vocab.items()}
    model_keys= list(model_avg_logits.keys())

    def objective(trial):
        raw    = [trial.suggest_float(f"w{j}", 0.0, 1.0) for j in range(len(model_keys))]
        total  = sum(raw) + 1e-9
        weights= [w / total for w in raw]

        n = len(next(iter(model_avg_logits.values())))
        preds: List[str] = []
        for i in range(n):
            combined: Optional[np.ndarray] = None
            for j, mname in enumerate(model_keys):
                lg = model_avg_logits[mname][i]
                if combined is None:
                    combined = weights[j] * lg.astype(np.float64)
                else:
                    t = min(combined.shape[0], lg.shape[0])
                    combined[:t] += weights[j] * lg[:t]
            preds.append(greedy_decode(torch.tensor(combined), id2token))

        score, _, _, _ = compute_score(val_df, preds)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_raw  = [study.best_params[f"w{j}"] for j in range(len(model_keys))]
    total     = sum(best_raw) + 1e-9
    best_w    = {mname: best_raw[j] / total for j, mname in enumerate(model_keys)}
    print(f"  Best ensemble weights (Optuna): {best_w}")
    return best_w


# ---------------------------------------------------------------------------
# Main inference / test mode entry point
# ---------------------------------------------------------------------------

def generate_predictions(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab     = load_or_build_vocab(args.data_dir)
    fe        = build_feature_extractor()
    ckpt_dir  = Path(args.checkpoint_dir)
    data_path = Path(args.data_dir)
    id2token  = {v: k for k, v in vocab.items()}

    # ---- Test CSV ----
    meta_dir = data_path / "metadata"
    for fname in ("test_phones.csv", "test.csv"):
        test_csv = meta_dir / fname
        if test_csv.exists():
            break
    else:
        raise FileNotFoundError(f"No test CSV in {meta_dir}")

    test_df = pd.read_csv(test_csv)
    if "transcript" not in test_df.columns:
        test_df["transcript"] = test_df.get("canonical", "")

    test_ds = MDDDataset(test_df, args.data_dir, vocab, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False,
        collate_fn=make_collate_fn(fe, device, spec_augment=False),
        num_workers=0,
    )

    # ---- Checkpoint discovery ----
    checkpoint_paths: Dict[str, List[str]] = {m: [] for m in BACKBONE_NAMES}
    for ckpt_file in sorted(ckpt_dir.glob("best_fold*.pt")):
        for mname in BACKBONE_NAMES:
            safe = mname.replace("/", "_").replace("-", "_")
            if safe in ckpt_file.name:
                checkpoint_paths[mname].append(str(ckpt_file))

    if not any(checkpoint_paths.values()):
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    vocab_size = len(vocab)

    # ---- Calibration: use last fold's val set ----
    temperatures: Dict[str, float] = {}
    val_df_cal: Optional[pd.DataFrame] = None
    val_loader_cal: Optional[DataLoader] = None
    try:
        splits = get_kfold_splits(args.data_dir, 5, 42)
        val_df_cal = splits[-1][1]
        val_ds_cal = MDDDataset(val_df_cal, args.data_dir, vocab, augment=False)
        val_loader_cal = DataLoader(
            val_ds_cal, batch_size=8, shuffle=False,
            collate_fn=make_collate_fn(fe, device, spec_augment=False),
            num_workers=0,
        )
    except Exception as e:
        print(f"  Warning: could not build val loader for calibration: {e}")

    for mname, ckpts in checkpoint_paths.items():
        if not ckpts or val_loader_cal is None:
            continue
        model = create_model(mname, vocab_size, device)
        model.load_state_dict(torch.load(ckpts[0], map_location=device))
        T = temperature_scale(model, val_loader_cal, vocab, device)
        temperatures[mname] = T
        del model
        torch.cuda.empty_cache()

    # ---- Collect test logits per model (average across folds) ----
    test_avg_logits: Dict[str, List[np.ndarray]] = {}
    for mname, ckpts in checkpoint_paths.items():
        if not ckpts:
            continue
        T_cal = temperatures.get(mname, 1.0)
        fold_logit_list: List[List[np.ndarray]] = []
        for ckpt in ckpts:
            model = create_model(mname, vocab_size, device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            fold_logits = _collect_logits(model, test_loader, device, temperature=T_cal)
            fold_logit_list.append(fold_logits)
            del model
            torch.cuda.empty_cache()

        # Average over folds
        n = len(fold_logit_list[0])
        test_avg_logits[mname] = [
            _avg_logits([fold[i] for fold in fold_logit_list]) for i in range(n)
        ]

    # ---- Optuna: find ensemble weights on validation ----
    ensemble_weights: Dict[str, float] = {}
    if val_df_cal is not None and val_loader_cal is not None:
        val_avg_logits: Dict[str, List[np.ndarray]] = {}
        for mname, ckpts in checkpoint_paths.items():
            if not ckpts:
                continue
            T_cal = temperatures.get(mname, 1.0)
            fold_logit_list = []
            for ckpt in ckpts:
                model = create_model(mname, vocab_size, device)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                fold_logits = _collect_logits(model, val_loader_cal, device, temperature=T_cal)
                fold_logit_list.append(fold_logits)
                del model
                torch.cuda.empty_cache()
            n = len(fold_logit_list[0])
            val_avg_logits[mname] = [
                _avg_logits([fl[i] for fl in fold_logit_list]) for i in range(n)
            ]

        ensemble_weights = find_ensemble_weights_optuna(
            val_avg_logits, val_df_cal, vocab, n_trials=100,
        )
    else:
        active = [m for m, cs in test_avg_logits.items() if cs]
        ensemble_weights = {m: 1.0 / len(active) for m in active}

    # ---- Final ensemble decode ----
    final_preds: List[str] = []
    n_test = len(test_df)
    for i in range(n_test):
        combined: Optional[np.ndarray] = None
        for mname, avg_list in test_avg_logits.items():
            w = ensemble_weights.get(mname, 0.0)
            if w <= 0 or i >= len(avg_list):
                continue
            lg = avg_list[i].astype(np.float64)
            if combined is None:
                combined = w * lg
            else:
                t = min(combined.shape[0], lg.shape[0])
                combined[:t] += w * lg[:t]
        hyp = greedy_decode(torch.tensor(combined if combined is not None else np.zeros((1, len(vocab)))), id2token)
        final_preds.append(hyp)

    out_df = pd.DataFrame({"id": range(len(final_preds)), "predict": final_preds})
    out_df.to_csv("predictions.csv", index=False)
    print(f"\nWrote {len(final_preds)} predictions → predictions.csv")
