from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from src.utils import PAD_TOKEN_ID, SAMPLE_RATE, text_to_tensor


# ---------------------------------------------------------------------------
# Audiomentations augmentation (lazy import to avoid hard dependency crash)
# ---------------------------------------------------------------------------

def _build_augmenter():
    try:
        from audiomentations import AddGaussianNoise, Compose, PitchShift, TimeStretch
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.2),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
        ])
    except ImportError:
        return None


_AUGMENTER = None


def _get_augmenter():
    global _AUGMENTER
    if _AUGMENTER is None:
        _AUGMENTER = _build_augmenter()
    return _AUGMENTER


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MDDDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        vocab: Dict[str, int],
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.augment = augment
        self.augmenter = _get_augmenter() if augment else None

        self.paths      = list(df["path"])
        self.canonicals = list(df["canonical"])
        self.transcripts= list(df["transcript"])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        wav_path = self.data_dir / self.paths[idx]
        waveform, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE)

        if self.augment and self.augmenter is not None:
            waveform = self.augmenter(samples=waveform, sample_rate=SAMPLE_RATE)
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)

        canonical_ids  = text_to_tensor(self.canonicals[idx],  self.vocab)
        transcript_ids = text_to_tensor(self.transcripts[idx], self.vocab)
        has_error = int(str(self.canonicals[idx]) != str(self.transcripts[idx]))

        return waveform, canonical_ids, transcript_ids, has_error


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def make_collate_fn(feature_extractor, device: torch.device, spec_augment: bool = False):
    # SpecAugment: time masking on 1-D extracted features
    # FrequencyMasking does not apply to raw waveforms; only TimeMasking is used.
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=100)

    def collate_fn(batch):
        with torch.no_grad():
            max_ling_len  = max(len(row[1]) for row in batch)
            max_trans_len = max(len(row[2]) for row in batch)

            waveforms:    List[np.ndarray] = []
            linguistics:  List[List[int]]  = []
            transcripts:  List[List[int]]  = []
            out_lengths:  List[int]        = []
            wav_lengths:  List[int]        = []
            error_labels: List[int]        = []

            for waveform, ling, trans, has_error in batch:
                wav_lengths.append(waveform.shape[0])
                waveforms.append(waveform)

                padded_ling = ling + [PAD_TOKEN_ID] * (max_ling_len - len(ling))
                linguistics.append(padded_ling)

                out_lengths.append(len(trans))
                padded_trans = trans + [PAD_TOKEN_ID] * (max_trans_len - len(trans))
                transcripts.append(padded_trans)

                error_labels.append(has_error)

            inputs = feature_extractor(waveforms, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            input_values = torch.nan_to_num(input_values, nan=0.0, posinf=1.0, neginf=-1.0)

            # SpecAugment: time masking on raw 1-D features
            if spec_augment:
                # unsqueeze to (B, 1, T): torchaudio treats last dim as time
                x = input_values.unsqueeze(1)
                x = time_mask(x)
                input_values = x.squeeze(1)

            linguistic_t  = torch.tensor(linguistics,  dtype=torch.long,  device=device)
            transcript_t  = torch.tensor(transcripts,  dtype=torch.long,  device=device)
            out_lengths_t = torch.tensor(out_lengths,  dtype=torch.long,  device=device)
            wav_lengths_t = torch.tensor(wav_lengths,  dtype=torch.long,  device=device)
            error_labels_t= torch.tensor(error_labels, dtype=torch.long,  device=device)

            return input_values, linguistic_t, transcript_t, out_lengths_t, wav_lengths_t, error_labels_t

    return collate_fn


# ---------------------------------------------------------------------------
# K-fold split
# ---------------------------------------------------------------------------

def get_kfold_splits(
    data_dir: str,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    csv_path = Path(data_dir) / "metadata" / "train_phones.csv"
    df = pd.read_csv(csv_path)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        splits.append((train_df, val_df))
    return splits
