#!/usr/bin/env python3
"""Stage 1 preprocessing v2: aligned windows, tau_p feature, configurable normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

DATA_ROOT = Path("/Users/keremciftci/Desktop/KO_ARMT_data")
ARTIFACT_ROOT = DATA_ROOT / "artifacts"
ARTIFACT_ROOT.mkdir(exist_ok=True)


def load_events() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "processed_events.csv", dtype={"eq_code": str})


def load_timeseries() -> tuple[np.memmap, list[str]]:
    traces = np.load(DATA_ROOT / "z_traces.npy", mmap_mode="r")
    eq_df = pd.read_csv(DATA_ROOT / "z_eq_codes.csv", dtype={"eq_code": str})
    return traces, eq_df["eq_code"].tolist()


def parse_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def compute_tau_samples(
    row: pd.Series,
    method: Literal["travel_time", "sta_lta", "null"],
    vp_km_per_s: float,
    sampling_rate: float,
) -> int:
    if method == "null":
        return 0
    if method == "travel_time":
        distance = float(row["distance_km"])
        travel_time = distance / max(vp_km_per_s, 0.1)
        return int(round(travel_time * sampling_rate))
    raise NotImplementedError("STA/LTA picks to be implemented later.")


def global_stats(arrays: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.concatenate(arrays)
    mean = float(stacked.mean())
    std = float(stacked.std())
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Prepare Stage 1 v2 artifacts.")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML (e.g., configs/stage1_v2.yaml)")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--output-version", type=str, default="v2")
    args = parser.parse_args()

    config = parse_config(args.config)
    dataset_cfg = config["dataset"]
    version = dataset_cfg.get("version", "v2")
    norm_mode = dataset_cfg.get("norm_mode", "global")
    add_tau = dataset_cfg.get("add_tau_p", True)
    tau_method = dataset_cfg.get("tau_method", "travel_time")
    vp = dataset_cfg.get("vp_km_per_s", 6.0)

    window_cfg = dataset_cfg.get("window", {})
    align_on_pick = window_cfg.get("align_on_pick", True)
    length_seconds = window_cfg.get("length_seconds", 40.0)
    pre_seconds = window_cfg.get("pre_seconds", 5.0)
    length_samples = int(round(length_seconds * args.sampling_rate))
    pre_samples = int(round(pre_seconds * args.sampling_rate))

    traces, eq_codes = load_timeseries()
    events = load_events().set_index("eq_code")

    arrays = []
    cond_rows = []
    picks = []

    for idx, code in enumerate(eq_codes):
        if code not in events.index:
            continue
        row = events.loc[code]
        trace = np.asarray(traces[idx], dtype=np.float32)
        tau = compute_tau_samples(row, tau_method, vp, args.sampling_rate) if add_tau else 0

        if align_on_pick:
            start = max(tau - pre_samples, 0)
            end = start + length_samples
            if end > trace.shape[0]:
                start = max(trace.shape[0] - length_samples, 0)
                end = start + length_samples
            window = trace[start:end]
            if window.shape[0] < length_samples:
                pad = length_samples - window.shape[0]
                window = np.pad(window, (0, pad), mode="constant")
        else:
            window = trace[:length_samples]

        arrays.append(window)

        cond = {
            "eq_code": code,
            "lat": row["lat"],
            "lon": row["lon"],
            "depth": row["depth"],
            "mag": row["mag"],
            "distance_km": row["distance_km"],
            "azimuth": row["azimuth"],
        }
        if add_tau:
            cond["tau_samples"] = tau
        cond["log_rms"] = float(np.log(np.sqrt(np.mean(window**2)) + 1e-8))
        cond_rows.append(cond)

        picks.append(
            {
                "eq_code": code,
                "tau_samples": tau,
                "window_start": int(start if align_on_pick else 0),
                "window_length": int(length_samples),
                "tau_method": tau_method,
            }
        )

    arrays = np.stack(arrays).astype(np.float32)

    output_dir = ARTIFACT_ROOT / version
    output_dir.mkdir(exist_ok=True, parents=True)

    if norm_mode == "global":
        mean, std = global_stats(list(arrays))
        arrays_norm = (arrays - mean) / max(std, 1e-6)
        norm_stats = {"mean": mean, "std": std, "mode": "global"}
    elif norm_mode == "per_trace":
        means = arrays.mean(axis=1, keepdims=True)
        stds = arrays.std(axis=1, keepdims=True)
        arrays_norm = (arrays - means) / np.maximum(stds, 1e-6)
        norm_stats = {"mode": "per_trace"}
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    traces_path = output_dir / "z_traces.npy"
    np.save(traces_path, arrays_norm)

    cond_df = pd.DataFrame(cond_rows)
    az = np.deg2rad(cond_df["azimuth"])
    cond_df["az_sin"] = np.sin(az)
    cond_df["az_cos"] = np.cos(az)
    eq_codes_series = cond_df["eq_code"].copy()
    cond_numeric = cond_df.drop(columns=["azimuth", "eq_code"])
    cond_np = cond_numeric.to_numpy(dtype=np.float32)
    cond_numeric.to_csv(output_dir / "z_conditions.csv", index=False)
    np.save(output_dir / "z_conditions.npy", cond_np)

    norm_path = output_dir / "norm_stats.json"
    norm_path.write_text(json.dumps(norm_stats, indent=2))

    columns_path = output_dir / "z_condition_columns.json"
    columns_path.write_text(json.dumps(cond_numeric.columns.tolist(), indent=2))

    eq_codes_path = output_dir / "z_eq_codes.csv"
    pd.DataFrame({"eq_code": eq_codes_series}).to_csv(eq_codes_path, index=False)

    pd.DataFrame(picks).to_csv(output_dir / "picks.csv", index=False)

    print(f"[prepare v2] wrote traces to {traces_path} with shape {arrays_norm.shape}")
    print(f"[prepare v2] wrote conditions to {output_dir / 'z_conditions.csv'}")
    print(f"[prepare v2] wrote condition columns to {columns_path}")
    print(f"[prepare v2] norm stats -> {norm_path}")
    print(f"[prepare v2] picks -> {output_dir / 'picks.csv'}")


if __name__ == "__main__":
    main()

