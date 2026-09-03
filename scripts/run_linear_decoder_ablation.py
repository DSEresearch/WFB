from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ReLU-MLP and linear decoders for WFB trajectory prediction.")
    parser.add_argument("--processed_dir", default="outputs/preprocessed_irregular")
    parser.add_argument("--output_dir", default="outputs/linear_decoder_ablation")
    parser.add_argument("--seeds", nargs="+", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--epochs", default="100")
    parser.add_argument("--patience", default="15")
    parser.add_argument("--batch_size", default="512")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "scripts/run_experiment.py",
        "--processed_dir",
        args.processed_dir,
        "--output_dir",
        args.output_dir,
        "--models",
        "ffn",
        "wfb_real_t",
        "--wfb_variants",
        "standard",
        "combined",
        "--lambda_laplacians",
        "1e-5",
        "--wfb_decoders",
        "mlp",
        "linear",
        "--seeds",
        *args.seeds,
        "--epochs",
        args.epochs,
        "--patience",
        args.patience,
        "--batch_size",
        args.batch_size,
        "--device",
        args.device,
    ]
    if args.num_workers is not None:
        command.extend(["--num_workers", args.num_workers])
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
