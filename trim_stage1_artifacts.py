#!/usr/bin/env python3
"""Trim flat/noise prefixes from Stage 1 artifacts and emit a new version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim flat prefixes from Stage 1 artifacts.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/v2"),
        help="Directory containing source artifacts (z_traces.npy, z_conditions.npy, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v3_trimmed"),
        help="Directory to write trimmed artifacts",
    )
    parser.add_argument("--sampling-rate", type=float, default=100.0, help="Sampling rate in Hz")
    parser.add_argument(
        "--rms-window",
        type=float,
        default=0.3,
        help="Window (seconds) for moving RMS envelope",
    )
    parser.add_argument(
        "--noise-window",
        type=float,
        default=5.0,
        help="Duration (seconds) of early window used to estimate noise level",
    )
    parser.add_argument(
        "--threshold-mult",
        type=float,
        default=3.0,
        help="Multiplier applied to RMS noise level to detect first signal arrival",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Seconds kept before detected signal index",
    )
    parser.add_argument(
        "--max-cut-s",
        type=float,
        default=15.0,
        help="Maximum number of seconds allowed to be trimmed",
    )
    parser.add_argument(
        "--cut-quantile",
        type=float,
        default=0.75,
        help="Quantile of per-trace cut predictions to apply as global trim",
    )
    parser.add_argument(
        "--min-tail-s",
        type=float,
        default=10.0,
        help="Ensure at least this many seconds remain after trimming",
    )
    parser.add_argument(
        "--force-cut-samples",
        type=int,
        default=None,
        help="Override computed cut with a fixed number of samples",
    )
    return parser.parse_args()


def moving_rms(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    window_samples = max(int(round(window_s * fs)), 1)
    if window_samples % 2 == 0:
        window_samples += 1
    pad = window_samples // 2
    pad_mode = "reflect" if x.size > 1 else "edge"
    padded = np.pad(x, (pad, pad), mode=pad_mode)
    kernel = np.ones(window_samples, dtype=np.float32) / float(window_samples)
    rms_sq = np.convolve(padded * padded, kernel, mode="valid")
    return np.sqrt(np.maximum(rms_sq, 0.0)).astype(np.float32)


def detect_flat_prefix_samples(
    x: np.ndarray,
    fs: float,
    rms_win_s: float,
    noise_win_s: float,
    thresh_mult: float,
    margin_s: float,
    max_cut_s: float,
) -> int:
    rms = moving_rms(x, fs, rms_win_s)
    if rms.size == 0:
        return 0
    noise_samples = max(int(round(noise_win_s * fs)), 1)
    noise_slice = rms[: noise_samples] if noise_samples < rms.size else rms
    noise_level = float(noise_slice.std())
    threshold = noise_level * thresh_mult
    signal_idx = np.where(rms > threshold)[0]
    if signal_idx.size == 0:
        return 0
    first_sig = int(signal_idx[0])
    margin = int(round(margin_s * fs))
    cut_samples = max(0, first_sig - margin)
    max_cut = int(round(max_cut_s * fs))
    return int(min(cut_samples, max_cut))


def ensure_monotonic_columns(cond_columns: Optional[list[str]], cond_array: np.ndarray) -> pd.DataFrame:
    if cond_columns is None:
        raise ValueError("Condition columns metadata is required.")
    if len(cond_columns) != cond_array.shape[1]:
        raise ValueError("Condition columns metadata length mismatch.")
    return pd.DataFrame(cond_array, columns=cond_columns)


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    traces_path = source_dir / "z_traces.npy"
    conds_path = source_dir / "z_conditions.npy"
    cond_cols_path = source_dir / "z_condition_columns.json"
    eq_codes_path = source_dir / "z_eq_codes.csv"
    norm_stats_path = source_dir / "norm_stats.json"
    picks_path = source_dir / "picks.csv"

    if not traces_path.exists():
        raise FileNotFoundError(traces_path)
    if not conds_path.exists():
        raise FileNotFoundError(conds_path)
    if not cond_cols_path.exists():
        raise FileNotFoundError(cond_cols_path)
    if not eq_codes_path.exists():
        raise FileNotFoundError(eq_codes_path)

    traces_norm = np.load(traces_path, mmap_mode="r")
    cond_array = np.load(conds_path)
    with cond_cols_path.open() as f:
        cond_columns = json.load(f)
    cond_df = ensure_monotonic_columns(cond_columns, cond_array)
    eq_codes_df = pd.read_csv(eq_codes_path, dtype={"eq_code": str})

    norm_stats = None
    if norm_stats_path.exists():
        norm_stats = json.loads(norm_stats_path.read_text())

    if norm_stats is None or norm_stats.get("mode", "global") != "global":
        raise RuntimeError("This trimming script currently expects global normalization stats.")

    mean_old = float(norm_stats.get("mean", 0.0))
    std_old = float(norm_stats.get("std", 1.0))
    if std_old <= 0.0:
        raise RuntimeError("Invalid source normalization standard deviation (<= 0).")

    fs = float(args.sampling_rate)

    raw_windows: list[np.ndarray] = []
    flat_cuts: list[int] = []

    for idx in range(traces_norm.shape[0]):
        trace_norm = np.asarray(traces_norm[idx], dtype=np.float32)
        trace_raw = trace_norm * std_old + mean_old
        cut = detect_flat_prefix_samples(
            trace_raw,
            fs=fs,
            rms_win_s=args.rms_window,
            noise_win_s=args.noise_window,
            thresh_mult=args.threshold_mult,
            margin_s=args.margin,
            max_cut_s=args.max_cut_s,
        )
        flat_cuts.append(cut)
        raw_windows.append(trace_raw)

    flat_cuts_arr = np.array(flat_cuts, dtype=np.int32)
    cut_samples: int
    if args.force_cut_samples is not None:
        cut_samples = max(0, int(args.force_cut_samples))
    else:
        quantile = float(np.clip(args.cut_quantile, 0.0, 1.0))
        cut_estimate = float(np.quantile(flat_cuts_arr, quantile))
        cut_samples = int(round(cut_estimate))

    max_allowed_cut = raw_windows[0].shape[0] - int(round(args.min_tail_s * fs))
    if max_allowed_cut < 0:
        max_allowed_cut = raw_windows[0].shape[0] - 1
    if max_allowed_cut < 0:
        max_allowed_cut = 0
    cut_samples = max(0, min(cut_samples, max_allowed_cut))

    print(f"Using global cut of {cut_samples} samples ({cut_samples / fs:.2f} s)")

    trimmed_raw = [trace[cut_samples:] for trace in raw_windows]
    trimmed_length = trimmed_raw[0].shape[0]
    if trimmed_length <= 0:
        raise RuntimeError("Trimmed trace length is non-positive. Reduce cut parameters.")

    log_rms_values = [float(np.log(np.sqrt(np.mean(trace ** 2)) + 1e-8)) for trace in trimmed_raw]

    if "tau_samples" in cond_df.columns:
        cond_df["tau_samples"] = (cond_df["tau_samples"] - cut_samples).clip(lower=0.0)
    if "tau_seconds" in cond_df.columns:
        cond_df["tau_seconds"] = cond_df["tau_samples"] / fs
    if "log_rms" in cond_df.columns:
        cond_df["log_rms"] = log_rms_values

    trimmed_raw_array = np.stack(trimmed_raw).astype(np.float32)
    mean_new = float(trimmed_raw_array.mean())
    std_new = float(trimmed_raw_array.std())
    if std_new <= 1e-6:
        std_new = 1e-6
    trimmed_norm_array = (trimmed_raw_array - mean_new) / std_new

    np.save(output_dir / "z_traces.npy", trimmed_norm_array)
    cond_df.to_csv(output_dir / "z_conditions.csv", index=False)
    np.save(output_dir / "z_conditions.npy", cond_df.to_numpy(dtype=np.float32))
    cond_cols_path_out = output_dir / "z_condition_columns.json"
    cond_cols_path_out.write_text(json.dumps(list(cond_df.columns), indent=2))
    eq_codes_df.to_csv(output_dir / "z_eq_codes.csv", index=False)

    norm_stats_out = {
        "mode": "global",
        "mean": mean_new,
        "std": std_new,
        "source_norm_mean": mean_old,
        "source_norm_std": std_old,
        "cut_samples": int(cut_samples),
        "cut_seconds": float(cut_samples) / fs,
        "cut_quantile": float(args.cut_quantile) if args.force_cut_samples is None else None,
        "sampling_rate": fs,
    }
    (output_dir / "norm_stats.json").write_text(json.dumps(norm_stats_out, indent=2))

    trim_stats = {
        "min_cut": int(flat_cuts_arr.min()),
        "max_cut": int(flat_cuts_arr.max()),
        "mean_cut": float(flat_cuts_arr.mean()),
        "median_cut": float(np.median(flat_cuts_arr)),
        "chosen_cut_samples": int(cut_samples),
        "chosen_cut_seconds": float(cut_samples) / fs,
        "trimmed_length_samples": int(trimmed_length),
        "trimmed_length_seconds": float(trimmed_length) / fs,
    }
    (output_dir / "trim_stats.json").write_text(json.dumps(trim_stats, indent=2))

    if picks_path.exists():
        picks_df = pd.read_csv(picks_path)
        if "tau_samples" in picks_df.columns:
            picks_df["tau_samples"] = (
                picks_df["tau_samples"].astype(np.int64) - cut_samples
            ).clip(lower=0)
        if "window_start" in picks_df.columns:
            picks_df["window_start"] = picks_df["window_start"].astype(np.int64) + cut_samples
        if "window_length" in picks_df.columns:
            picks_df["window_length"] = int(trimmed_length)
        picks_df.to_csv(output_dir / "picks.csv", index=False)

    print(f"Trimmed artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
