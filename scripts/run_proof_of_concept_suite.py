from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = PROJECT_ROOT / "scripts" / "run_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the feed-forward WFB proof-of-concept controls; no sequence model is included."
    )
    parser.add_argument("--processed_dir", default="outputs/preprocessed_irregular")
    parser.add_argument("--output_dir", default="outputs/proof_of_concept_review")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=["position_controls", "motion_controls", "lambda_selection", "decoder"],
        default=["position_controls", "motion_controls", "lambda_selection", "decoder"],
    )
    parser.add_argument("--seeds", nargs="+", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--epochs", default="100")
    parser.add_argument("--patience", default="15")
    parser.add_argument("--batch_size", default="512")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", default=None)
    return parser.parse_args()


def common_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(RUN_EXPERIMENT),
        "--processed_dir",
        str(Path(args.processed_dir)),
        "--output_dir",
        str(output_dir),
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
    return command


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    control_models = [
        "ffn",
        "ffn_dt",
        "ffn_matched",
        "ffn_dt_matched",
        "sine_ffn_dt_matched",
        "wfb_real_t",
        "wfb_shuffled_t",
        "wfb_constant_t",
    ]

    if "position_controls" in args.suites:
        run(
            common_command(args, output_root / "position_controls")
            + [
                "--models",
                *control_models,
                "--input_features",
                "position",
                "--wfb_variants",
                "standard",
            ]
        )

    if "motion_controls" in args.suites:
        # Recomputing v/a after an interval intervention prevents real-dt motion
        # features from leaking into shuffled and constant controls.
        run(
            common_command(args, output_root / "motion_controls_recomputed")
            + [
                "--models",
                *control_models,
                "--input_features",
                "motion",
                "--motion_dt_policy",
                "recompute",
                "--wfb_variants",
                "standard",
            ]
        )

    if "lambda_selection" in args.suites:
        run(
            common_command(args, output_root / "lambda_selection")
            + [
                "--models",
                "wfb_real_t",
                "--wfb_variants",
                "standard",
                "laplacian",
                "combined",
                "--lambda_laplacians",
                "1e-7",
                "1e-6",
                "1e-5",
                "1e-4",
                "1e-3",
            ]
        )

    if "decoder" in args.suites:
        run(
            common_command(args, output_root / "decoder")
            + [
                "--models",
                "ffn",
                "wfb_real_t",
                "--wfb_variants",
                "standard",
                "--wfb_decoders",
                "mlp",
                "linear",
            ]
        )


if __name__ == "__main__":
    main()
