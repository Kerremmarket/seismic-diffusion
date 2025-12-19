#!/usr/bin/env python3
"""Stage 1 data inspection for KO_ARMT seismic dataset."""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import pandas as pd


DATA_ROOT = Path("/Users/keremciftci/Desktop/KO_ARMT_data")


def inspect_events() -> set[str]:
    """Inspect processed_events.csv and report basic stats."""
    events_path = DATA_ROOT / "processed_events.csv"
    events = pd.read_csv(events_path)

    print("\n=== processed_events.csv ===")
    print(f"rows: {len(events)}")
    stations = events["station"].unique().tolist()
    print(f"stations: {stations}")

    dupes = int(events["eq_code"].duplicated().sum())
    if dupes:
        print(f"duplicate eq_code count: {dupes}")
    else:
        print("duplicate eq_code count: 0")

    missing = events.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("missing values:\n", missing)
    else:
        print("missing values: none")

    print("\nSummary statistics:")
    print(events.describe(include="all"))

    return set(events["eq_code"].astype(str))


def inspect_timeseries(
    csv_path: Path, preview_rows: int = 3
) -> tuple[set[str], int, int, int]:
    """Inspect timeseries CSV and output basic stats."""
    print(f"\n=== {csv_path.name} ===")
    with csv_path.open("r", newline="") as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader)
        trace_start = next(
            (i for i, name in enumerate(header) if name.startswith("t")),
            None,
        )
        if trace_start is None:
            raise RuntimeError("Could not locate trace columns starting with 't'")

        meta_columns = header[:trace_start]
        trace_len = len(header) - trace_start
        print(
            f"columns: {len(header)} "
            f"(metadata={len(meta_columns)}, trace length={trace_len})"
        )
        print("metadata columns (first 10):", meta_columns[:10])

        eq_codes: set[str] = set()
        lengths: set[int] = set()
        previews: list[dict[str, float | str | int]] = []
        row_count = 0
        sampling_rates: set[float] = set()

        for row in reader:
            row_count += 1
            eq_code = row[0]
            eq_codes.add(eq_code)
            lengths.add(len(row) - trace_start)

            try:
                sr_idx = meta_columns.index("SamplingRate")
                sampling_rates.add(float(row[sr_idx]))
            except ValueError:
                pass

            if len(previews) < preview_rows:
                samples = np.asarray(row[trace_start:], dtype=np.float32)
                previews.append(
                    {
                        "eq_code": eq_code,
                        "trace_len": samples.size,
                        "mean": float(samples.mean()),
                        "std": float(samples.std()),
                        "min": float(samples.min()),
                        "max": float(samples.max()),
                    }
                )

        print(f"rows: {row_count}")
        print(f"distinct trace lengths: {sorted(lengths)}")
        if sampling_rates:
            print(f"sampling rates observed: {sorted(sampling_rates)}")
        print("preview stats:")
        for item in previews:
            print("  ", item)

    return eq_codes, trace_len, row_count, trace_start


def main() -> None:
    event_codes = inspect_events()

    ts_codes, trace_len, row_count, trace_start = inspect_timeseries(
        DATA_ROOT / "timeseries_Z.csv"
    )

    missing_in_ts = event_codes - ts_codes
    missing_in_events = ts_codes - event_codes
    print("\n=== Cross-check ===")
    print(f"events without Z trace: {len(missing_in_ts)}")
    print(f"Z traces without metadata: {len(missing_in_events)}")
    if missing_in_ts:
        print("sample:", list(sorted(missing_in_ts))[:10])
    if missing_in_events:
        print("sample:", list(sorted(missing_in_events))[:10])

    print(
        f"\nTrace length: {trace_len}, metadata columns: {trace_start}, "
        f"total traces: {row_count}"
    )


if __name__ == "__main__":
    main()


