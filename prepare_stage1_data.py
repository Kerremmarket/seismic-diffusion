#!/usr/bin/env python3
"""Stage 1 preprocessing for KO_ARMT seismic dataset.

Converts timeseries_Z.csv to a memory-mapped NumPy array and builds normalized
conditioning vectors from processed_events.csv.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd


DATA_ROOT = Path("/Users/keremciftci/Desktop/KO_ARMT_data")
Z_CSV = DATA_ROOT / "timeseries_Z.csv"
EVENTS_CSV = DATA_ROOT / "processed_events.csv"

TRACES_NPY = DATA_ROOT / "z_traces.npy"
EQ_INDEX_CSV = DATA_ROOT / "z_eq_codes.csv"
COND_NPY = DATA_ROOT / "z_conditions.npy"
COND_STATS_JSON = DATA_ROOT / "z_condition_norm.json"


def determine_shape(csv_path: Path) -> tuple[int, int, int]:
    """Determine number of rows, trace length, and metadata column count."""
    with csv_path.open("r", newline="") as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader)
        trace_start = next(
            (i for i, name in enumerate(header) if name.startswith("t")),
            None,
        )
        if trace_start is None:
            raise RuntimeError("Could not locate trace columns starting with 't'")
        trace_len = len(header) - trace_start
        row_count = sum(1 for _ in reader)
    return row_count, trace_len, trace_start


def convert_csv_to_memmap(
    csv_path: Path,
    npy_path: Path,
    index_csv_path: Path,
    dtype: np.dtype[np.float32] = np.float32,
) -> tuple[int, int, int]:
    """Convert the waveform CSV into a float32 memmap and save eq_code ordering."""
    print("Pass 1: determining shape...")
    row_count, trace_len, trace_start = determine_shape(csv_path)
    print(
        f"rows={row_count}, trace_len={trace_len}, "
        f"metadata_columns={trace_start}"
    )

    mmap = np.lib.format.open_memmap(
        npy_path,
        mode="w+",
        dtype=dtype,
        shape=(row_count, trace_len),
    )
    eq_codes: list[str] = []

    print("Pass 2: writing memmap...")
    with csv_path.open("r", newline="") as file_obj:
        reader = csv.reader(file_obj)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            eq_code = row[0]
            eq_codes.append(eq_code)
            trace = np.asarray(row[trace_start:], dtype=dtype)
            if trace.size != trace_len:
                raise ValueError(
                    f"inconsistent trace length for {eq_code}: {trace.size} vs {trace_len}"
                )
            mmap[i, :] = trace
            if (i + 1) % 100 == 0 or (i + 1) == row_count:
                print(f"  processed {i + 1}/{row_count}", end="\r")

    mmap.flush()
    print("\nmemmap saved:", npy_path)

    pd.DataFrame({"eq_code": eq_codes}).to_csv(index_csv_path, index=False)
    print("eq_code index saved:", index_csv_path)

    return row_count, trace_len, trace_start


def build_conditions(
    events_csv: Path,
    index_csv: Path,
    cond_npy_path: Path,
    stats_json_path: Path,
) -> None:
    """Build normalized conditioning vectors aligned with waveform memmap."""
    events = pd.read_csv(events_csv, dtype={"eq_code": str})
    index_df = pd.read_csv(index_csv, dtype={"eq_code": str})

    merged = index_df.merge(events, on="eq_code", how="inner")
    if len(merged) != len(index_df):
        missing = sorted(set(index_df["eq_code"]) - set(merged["eq_code"]))
        raise RuntimeError(f"{len(missing)} traces missing metadata: {missing[:10]}")

    cond = pd.DataFrame(index=merged.index)
    cond["lat"] = merged["lat"]
    cond["lon"] = merged["lon"]
    cond["depth"] = merged["depth"]
    cond["mag"] = merged["mag"]
    cond["distance_km"] = merged["distance_km"]

    azimuth_rad = np.deg2rad(merged["azimuth"].to_numpy())
    cond["az_sin"] = np.sin(azimuth_rad)
    cond["az_cos"] = np.cos(azimuth_rad)

    mean = cond.mean()
    std = cond.std().replace(0, 1.0)
    cond_norm = ((cond - mean) / std).astype(np.float32)

    np.save(cond_npy_path, cond_norm.to_numpy())
    print("conditions saved:", cond_npy_path)

    stats = {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "ordering": cond.columns.tolist(),
    }
    stats_json_path.write_text(json.dumps(stats, indent=2))
    print("condition normalization stats saved:", stats_json_path)


def main() -> None:
    row_count, trace_len, trace_start = convert_csv_to_memmap(
        Z_CSV, TRACES_NPY, EQ_INDEX_CSV
    )
    build_conditions(EVENTS_CSV, EQ_INDEX_CSV, COND_NPY, COND_STATS_JSON)
    print(
        "\nDone. Artifacts:"
        f"\n  {TRACES_NPY} (shape: {row_count} x {trace_len})"
        f"\n  metadata columns skipped: {trace_start}"
        f"\n  {EQ_INDEX_CSV}"
        f"\n  {COND_NPY}"
        f"\n  {COND_STATS_JSON}"
    )


if __name__ == "__main__":
    main()


