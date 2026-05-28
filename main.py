import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDD Challenge 2025 — Vietnamese Mispronunciation Detection and Diagnosis"
    )
    parser.add_argument(
        "--mode", choices=["eval", "test"], required=True,
        help="'eval': 5-fold CV on training data. 'test': inference on unlabelled test set.",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root of the dataset (e.g. ./data/MDD-Challenge-2025-training-set).",
    )
    # Training hyperparameters
    parser.add_argument("--n_folds",      type=int,   default=5,    help="Number of CV folds.")
    parser.add_argument("--epochs",       type=int,   default=30,   help="Training epochs per fold.")
    parser.add_argument("--batch_size",   type=int,   default=16,   help="Training batch size.")
    parser.add_argument("--lr",           type=float, default=1e-4, help="Base learning rate.")
    parser.add_argument("--llrd_decay",   type=float, default=0.9,  help="LLRD decay factor.")
    parser.add_argument("--focal_gamma",  type=float, default=2.0,  help="Focal loss gamma.")
    parser.add_argument("--seed",         type=int,   default=42,   help="Random seed.")
    # Inference
    parser.add_argument(
        "--checkpoint_dir", default="./outputs/checkpoints",
        help="Directory containing trained checkpoints (used for --mode test).",
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
