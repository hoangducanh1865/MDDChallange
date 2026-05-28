import os
import argparse

# Enable CPU fallback for MPS (Apple Silicon) on ops not yet supported by Metal.
# Has no effect on CUDA or CPU-only machines.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import transformers
# Suppress "UNEXPECTED / MISSING key" load reports from from_pretrained().
# We intentionally load ForCTC checkpoints into the base Model class,
# so the mismatched keys are expected and safe to ignore.
transformers.logging.set_verbosity_warning()


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDD Challenge 2025 — Vietnamese Mispronunciation Detection and Diagnosis"
    )
    parser.add_argument(
        "--mode", choices=["eval", "test"], required=True,
        help="'eval': 5-fold CV on training data. 'test': inference on unlabelled test set.",
    )
    parser.add_argument(
        "--data_dir", default="./data/MDD-Challenge-2025-training-set",
        help="Root of the dataset (e.g. ./data/MDD-Challenge-2025-training-set).",
    )
    # Model selection
    parser.add_argument(
        "--model", default="facebook/wav2vec2-base-100h",
        help=(
            "Backbone model to train. Supported values:\n"
            "  facebook/wav2vec2-base-100h (default)\n"
            "  vinai/wav2vec2-base-vietnamese-250h\n"
            "  hubert-base-ls960  (local, requires models/ folder)"
        ),
    )
    # Training hyperparameters
    parser.add_argument("--n_folds",      type=int,   default=5,    help="Number of CV folds.")
    parser.add_argument("--epochs",       type=int,   default=30,   help="Training epochs per fold.")
    parser.add_argument("--batch_size",   type=int,   default=16,   help="Training batch size.")
    parser.add_argument("--lr",           type=float, default=1e-4, help="Base learning rate.")
    parser.add_argument("--llrd_decay",   type=float, default=0.9,  help="LLRD decay factor.")
    parser.add_argument("--focal_gamma",  type=float, default=2.0,  help="Focal loss gamma.")
    parser.add_argument("--seed",         type=int,   default=42,   help="Random seed.")
    # Checkpoint / inference
    parser.add_argument(
        "--checkpoint_dir", default="./outputs/checkpoints",
        help=(
            "Root checkpoint directory. Checkpoints are saved to "
            "<checkpoint_dir>/<model_name>/fold{i}_epoch{j}.pt and fold{i}_best.pt. "
            "For --mode test, all subdirectories are scanned automatically."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "eval":
        from src.train import run_cross_validation
        run_cross_validation(args)
    else:
        from src.inference import generate_predictions
        generate_predictions(args)


if __name__ == "__main__":
    main()
