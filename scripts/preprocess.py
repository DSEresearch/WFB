from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import WindowConfig, add_motion_features, fit_normalization, load_raw_trajectories, make_windows, split_windows
try:
    from src.data import inspect_annotation_files
except ImportError:
    inspect_annotation_files = None
from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ETH/UCY/JAAD trajectories into fixed-length windows.")
    parser.add_argument("--data_dir", type=str, default="eth_ucy_jaad")
    parser.add_argument("--output_dir", type=str, default="outputs/preprocessed")
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=2.5)
    parser.add_argument("--max_track_gap", type=int, default=10, help="Maximum frame gap allowed when linking YOLO detections into tracks.")
    parser.add_argument("--max_link_distance", type=float, default=None, help="Optional max position distance for YOLO track linking.")
    parser.add_argument("--irregular_obs", action="store_true", help="Sample nonuniform observation frames to make delta_t informative.")
    parser.add_argument("--max_obs_skip", type=int, default=3, help="Maximum row/frame skip between observed points when --irregular_obs is enabled.")
    parser.add_argument("--irregular_samples", type=int, default=2, help="Number of irregular windows sampled per start index.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument(
        "--split_unit",
        choices=["source_agent", "source"],
        default="source_agent",
        help="Use source for a stricter scene/source-disjoint evaluation; source_agent preserves the paper split.",
    )
    parser.add_argument("--inspect_only", action="store_true", help="Print dataset file/column diagnostics and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    if args.inspect_only:
        if inspect_annotation_files is None:
            raise RuntimeError(
                "--inspect_only requires the updated src/data.py. Copy the latest src/data.py to the server, "
                "or run preprocessing without --inspect_only."
            )
        report = inspect_annotation_files(args.data_dir)
        write_json(out_dir / "dataset_inspection.json", report)
        print(report)
        return
    cfg = WindowConfig(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        stride=args.stride,
        fps=args.fps,
        irregular_obs=args.irregular_obs,
        max_obs_skip=args.max_obs_skip,
        irregular_samples=args.irregular_samples,
        seed=args.seed,
    )
    raw = load_raw_trajectories(
        args.data_dir,
        fps=args.fps,
        max_track_gap=args.max_track_gap,
        max_link_distance=args.max_link_distance,
    )
    features = add_motion_features(raw, min_dt=cfg.min_dt)
    windows = make_windows(features, cfg)
    splits = split_windows(
        windows,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        split_unit=args.split_unit,
    )
    norm = fit_normalization(splits["train"], obs_len=args.obs_len)
    write_json(out_dir / "normalization.json", norm)
    dt_cols = [f"obs_{i}_dt" for i in range(args.obs_len)]
    dt_values = windows[dt_cols].to_numpy(dtype=np.float32).reshape(-1)

    summary = {
        "data_dir": str(Path(args.data_dir)),
        "obs_len": args.obs_len,
        "pred_len": args.pred_len,
        "stride": args.stride,
        "fps": args.fps,
        "max_track_gap": args.max_track_gap,
        "max_link_distance": args.max_link_distance,
        "irregular_obs": args.irregular_obs,
        "max_obs_skip": args.max_obs_skip,
        "irregular_samples": args.irregular_samples,
        "split_unit": args.split_unit,
        "raw_rows": int(len(raw)),
        "agents": int(raw[["source", "agent_id"]].drop_duplicates().shape[0]),
        "windows": int(len(windows)),
        "split_windows": {name: int(len(df)) for name, df in splits.items()},
        "split_sources": {
            name: sorted(df["source"].astype(str).unique().tolist()) for name, df in splits.items()
        },
        "split_agents": {
            name: int(df[["source", "agent_id"]].drop_duplicates().shape[0]) for name, df in splits.items()
        },
        "delta_t": {
            "mean": float(dt_values.mean()),
            "std": float(dt_values.std()),
            "min": float(dt_values.min()),
            "max": float(dt_values.max()),
            "unique_rounded_first_20": sorted(np.unique(np.round(dt_values, 6)).tolist())[:20],
        },
    }
    for name, df in splits.items():
        df.to_csv(out_dir / f"windows_{name}.csv", index=False)
    write_json(out_dir / "preprocess_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
