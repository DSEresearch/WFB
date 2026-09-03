from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import TrajectoryWindowDataset
from src.models import build_model, count_parameters
from src.plotting import save_prediction_plot
from src.training import collect_predictions, evaluate, save_history, train_one_epoch
from src.utils import cpu_count_for_loader, ensure_dir, pick_device, set_seed, write_json


T_MODE_BY_MODEL = {
    "ffn": "real",
    "ffn_dt": "real",
    "ffn_matched": "real",
    "ffn_dt_matched": "real",
    "sine_ffn": "real",
    "sine_ffn_dt": "real",
    "sine_ffn_matched": "real",
    "sine_ffn_dt_matched": "real",
    "wfb_real_t": "real",
    "wfb_shuffled_t": "shuffled",
    "wfb_constant_t": "constant",
}
FEATURE_DIM_BY_SET = {
    "motion": 6,
    "position": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FFN and WFB-FFN trajectory predictors.")
    parser.add_argument("--processed_dir", type=str, default="outputs/preprocessed")
    parser.add_argument("--output_dir", type=str, default="outputs/runs")
    parser.add_argument("--models", nargs="+", default=["ffn", "wfb_real_t", "wfb_shuffled_t"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument(
        "--input_features",
        type=str,
        default="motion",
        choices=["motion", "position"],
        help="motion uses [x,y,vx,vy,ax,ay]; position uses only [x,y] to isolate delta_t effects.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--wave_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sine_omega_0", type=float, default=30.0)
    parser.add_argument(
        "--motion_dt_policy",
        choices=["fixed", "recompute"],
        default="fixed",
        help="For shuffled/constant controls, either retain real-dt motion features or recompute v/a with intervened dt.",
    )
    parser.add_argument("--wfb_variants", nargs="+", default=["standard"], choices=["standard", "laplacian", "combined"])
    parser.add_argument("--lambda_laplacians", nargs="+", type=float, default=[1e-5])
    parser.add_argument(
        "--wave_ablations",
        nargs="+",
        default=["full"],
        choices=["full", "without_A", "without_k", "without_omega", "without_theta"],
        help="Ablate WFB wave parameters. full uses A,k,omega,theta.",
    )
    parser.add_argument(
        "--wfb_decoders",
        nargs="+",
        default=["mlp"],
        choices=["mlp", "linear"],
        help="Decode WFB features with the existing ReLU MLP or a single linear output layer.",
    )
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def format_lambda(value: float) -> str:
    return f"{value:.0e}".replace("+", "")


def model_spec(model_name: str) -> tuple[str, bool, bool]:
    """Return (architecture, include_delta_t, parameter_matched)."""
    if model_name.startswith("wfb"):
        return "wfb", True, False
    matched = model_name.endswith("_matched")
    base_name = model_name.removesuffix("_matched")
    if base_name in {"ffn", "ffn_dt"}:
        return "ffn", base_name.endswith("_dt"), matched
    if base_name in {"sine_ffn", "sine_ffn_dt"}:
        return "sine_ffn", base_name.endswith("_dt"), matched
    raise ValueError(f"Unknown model name {model_name}. Valid: {sorted(T_MODE_BY_MODEL)}")


def closest_parameter_matched_width(
    args: argparse.Namespace,
    architecture: str,
    include_delta_t: bool,
    feature_dim: int,
) -> tuple[int, int]:
    """Choose the hidden width whose trainable count is closest to the WFB reference."""
    reference = build_model(
        "wfb",
        args.obs_len,
        args.pred_len,
        args.hidden_dim,
        args.wave_dim,
        args.depth,
        args.dropout,
        feature_dim=feature_dim,
        wfb_decoder="mlp",
    )
    target = count_parameters(reference)

    def parameters_at(width: int) -> int:
        candidate = build_model(
            architecture,
            args.obs_len,
            args.pred_len,
            width,
            args.wave_dim,
            args.depth,
            args.dropout,
            feature_dim=feature_dim,
            include_delta_t=include_delta_t,
            sine_omega_0=args.sine_omega_0,
        )
        return count_parameters(candidate)

    low, high = 1, 4096
    while low < high:
        middle = (low + high) // 2
        if parameters_at(middle) < target:
            low = middle + 1
        else:
            high = middle
    candidates = sorted({max(1, low - 1), low})
    width = min(candidates, key=lambda value: abs(parameters_at(value) - target))
    return width, target


def experiment_grid(args: argparse.Namespace) -> list[tuple[str, str, float, str, str, str]]:
    specs = []
    for model_name in args.models:
        architecture, _, _ = model_spec(model_name)
        if architecture != "wfb":
            specs.append((model_name, "standard", 0.0, "full", "mlp", model_name))
            continue
        for wave_ablation in args.wave_ablations:
            ablation_suffix = "" if wave_ablation == "full" else f"__{wave_ablation}"
            for decoder in args.wfb_decoders:
                decoder_suffix = "" if decoder == "mlp" else "__linear_decoder"
                for variant in args.wfb_variants:
                    if variant == "standard":
                        specs.append(
                            (
                                model_name,
                                variant,
                                0.0,
                                wave_ablation,
                                decoder,
                                f"{model_name}__STD-WFB-FFN{ablation_suffix}{decoder_suffix}",
                            )
                        )
                    else:
                        for lam in args.lambda_laplacians:
                            paper_name = "Laplacian-WFB-FFN" if variant == "laplacian" else "STD-Laplacian-WFB-FFN"
                            specs.append(
                                (
                                    model_name,
                                    variant,
                                    lam,
                                    wave_ablation,
                                    decoder,
                                    f"{model_name}__{paper_name}__lambda_{format_lambda(lam)}"
                                    f"{ablation_suffix}{decoder_suffix}",
                                )
                            )
    return specs


def make_loader(dataset: TrajectoryWindowDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def run_one(
    args: argparse.Namespace,
    model_name: str,
    seed: int,
    device: torch.device,
    wfb_variant: str,
    lambda_laplacian: float,
    wave_ablation: str,
    wfb_decoder: str,
    run_label: str,
) -> dict[str, object]:
    set_seed(seed)
    processed = Path(args.processed_dir)
    run_dir = ensure_dir(Path(args.output_dir) / run_label / f"seed_{seed}")
    norm_path = processed / "normalization.json"
    t_mode = T_MODE_BY_MODEL[model_name]
    workers = args.num_workers if args.num_workers is not None else cpu_count_for_loader(default=4)

    train_ds = TrajectoryWindowDataset(
        processed / "windows_train.csv",
        norm_path,
        args.obs_len,
        args.pred_len,
        t_mode=t_mode,
        input_features=args.input_features,
        t_seed=seed * 10_000 + 1,
        motion_dt_policy=args.motion_dt_policy,
    )
    val_ds = TrajectoryWindowDataset(
        processed / "windows_val.csv",
        norm_path,
        args.obs_len,
        args.pred_len,
        t_mode=t_mode,
        input_features=args.input_features,
        t_seed=seed * 10_000 + 2,
        motion_dt_policy=args.motion_dt_policy,
    )
    test_ds = TrajectoryWindowDataset(
        processed / "windows_test.csv",
        norm_path,
        args.obs_len,
        args.pred_len,
        t_mode=t_mode,
        input_features=args.input_features,
        t_seed=seed * 10_000 + 3,
        motion_dt_policy=args.motion_dt_policy,
    )
    train_loader = make_loader(train_ds, args.batch_size, True, workers)
    val_loader = make_loader(val_ds, args.batch_size, False, workers)
    test_loader = make_loader(test_ds, args.batch_size, False, workers)

    base_model, include_delta_t, parameter_matched = model_spec(model_name)
    effective_hidden_dim = args.hidden_dim
    matched_target_parameters = None
    if parameter_matched:
        effective_hidden_dim, matched_target_parameters = closest_parameter_matched_width(
            args,
            base_model,
            include_delta_t,
            FEATURE_DIM_BY_SET[args.input_features],
        )
    model = build_model(
        base_model,
        args.obs_len,
        args.pred_len,
        effective_hidden_dim,
        args.wave_dim,
        args.depth,
        args.dropout,
        feature_dim=FEATURE_DIM_BY_SET[args.input_features],
        wave_ablation=wave_ablation,
        wfb_decoder=wfb_decoder,
        include_delta_t=include_delta_t,
        sine_omega_0=args.sine_omega_0,
    ).to(device)
    parameter_count = count_parameters(model)
    write_json(
        run_dir / "run_config.json",
        {
            **vars(args),
            "model_name": model_name,
            "run_label": run_label,
            "architecture": base_model,
            "include_delta_t": include_delta_t,
            "t_mode": t_mode,
            "seed": seed,
            "effective_hidden_dim": effective_hidden_dim,
            "parameter_count": parameter_count,
            "matched_target_parameters": matched_target_parameters,
            "train_samples": len(train_ds),
            "validation_samples": len(val_ds),
            "test_samples": len(test_ds),
        },
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_laplacian,
            args.grad_clip,
            wfb_variant=wfb_variant,
            warmup=epoch <= args.warmup_epochs,
        )
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if val_metrics["ADE"] < best_val:
            best_val = val_metrics["ADE"]
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    preds, targets = collect_predictions(model, test_loader, device)
    save_prediction_plot(preds, targets, run_dir / "predictions.png")
    save_history(history, run_dir / "history.csv")
    result = {
        "model": run_label,
        "base_model": model_name,
        "architecture": base_model,
        "include_delta_t": include_delta_t,
        "t_mode": t_mode,
        "motion_dt_policy": args.motion_dt_policy,
        "parameter_matched": parameter_matched,
        "parameter_count": parameter_count,
        "matched_target_parameters": matched_target_parameters,
        "effective_hidden_dim": effective_hidden_dim,
        "wfb_variant": wfb_variant,
        "lambda_laplacian": lambda_laplacian,
        "wave_ablation": wave_ablation,
        "wfb_decoder": wfb_decoder,
        "input_features": args.input_features,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_ADE": best_val,
        **test_metrics,
    }
    write_json(run_dir / "metrics.json", result)
    return result


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    device = pick_device(args.device)
    write_json(
        Path(args.output_dir) / "experiment_manifest.json",
        {
            "arguments": vars(args),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
    )
    rows = []
    for model_name in args.models:
        if model_name not in T_MODE_BY_MODEL:
            raise ValueError(f"Unknown model name {model_name}. Valid: {sorted(T_MODE_BY_MODEL)}")
    for model_name, wfb_variant, lambda_laplacian, wave_ablation, wfb_decoder, run_label in experiment_grid(args):
        for seed in args.seeds:
            print(
                f"Running {run_label} seed={seed} variant={wfb_variant} "
                f"lambda={lambda_laplacian:g} ablation={wave_ablation} decoder={wfb_decoder} on {device}"
            )
            rows.append(
                run_one(
                    args,
                    model_name,
                    seed,
                    device,
                    wfb_variant,
                    lambda_laplacian,
                    wave_ablation,
                    wfb_decoder,
                    run_label,
                )
            )
    results = pd.DataFrame(rows)
    results.to_csv(Path(args.output_dir) / "results_by_seed.csv", index=False)
    metric_cols = ["ADE", "FDE", "MSE", "RMSE"]
    summary = results.groupby("model")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns.to_flat_index()]
    summary.to_csv(Path(args.output_dir) / "summary.csv", index=False)

    validation_summary = (
        results.groupby("model")[["best_val_ADE", *metric_cols]].agg(["mean", "std", "count"]).reset_index()
    )
    validation_summary.columns = ["_".join(col).rstrip("_") for col in validation_summary.columns.to_flat_index()]
    validation_summary.to_csv(Path(args.output_dir) / "validation_summary.csv", index=False)

    if "ffn" in set(results["model"]):
        paired_rows = []
        reference = results[results["model"] == "ffn"].set_index("seed")
        for model_label, model_rows in results[results["model"] != "ffn"].groupby("model"):
            comparison = model_rows.set_index("seed")
            common_seeds = sorted(set(reference.index).intersection(comparison.index))
            if not common_seeds:
                continue
            for metric in metric_cols:
                baseline_values = reference.loc[common_seeds, metric].to_numpy(dtype=float)
                model_values = comparison.loc[common_seeds, metric].to_numpy(dtype=float)
                differences = model_values - baseline_values
                improvements = 100.0 * (baseline_values - model_values) / baseline_values
                n = len(common_seeds)
                standard_error = float(differences.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
                paired_rows.append(
                    {
                        "reference": "ffn",
                        "model": model_label,
                        "metric": metric,
                        "paired_seeds": n,
                        "mean_difference_model_minus_reference": float(differences.mean()),
                        "difference_std": float(differences.std(ddof=1)) if n > 1 else np.nan,
                        "normal_approx_ci95_low": float(differences.mean() - 1.96 * standard_error),
                        "normal_approx_ci95_high": float(differences.mean() + 1.96 * standard_error),
                        "mean_relative_improvement_percent": float(improvements.mean()),
                    }
                )
        pd.DataFrame(paired_rows).to_csv(Path(args.output_dir) / "paired_comparisons_vs_ffn.csv", index=False)

    correction_rows = results[results["wfb_variant"].isin(["laplacian", "combined"])]
    if not correction_rows.empty:
        group_cols = ["base_model", "wfb_variant", "wave_ablation", "wfb_decoder", "input_features"]
        candidates = (
            correction_rows.groupby([*group_cols, "lambda_laplacian"], dropna=False)[
                ["best_val_ADE", *metric_cols]
            ]
            .mean()
            .reset_index()
        )
        selected_indices = candidates.groupby(group_cols, dropna=False)["best_val_ADE"].idxmin()
        selected = candidates.loc[selected_indices].sort_values(group_cols)
        selected.to_csv(Path(args.output_dir) / "lambda_selected_by_validation.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
