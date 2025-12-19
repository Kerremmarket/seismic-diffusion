#!/usr/bin/env python3
"""PyTorch dataset/dataloader utilities for Stage 1 seismic diffusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DATA_ROOT = Path("/Users/keremciftci/Desktop/KO_ARMT_data")


class SeismicZDataset(Dataset):
    """Streams Z-component traces and conditional features from memmapped arrays."""

    def __init__(
        self,
        data_root: Path,
        slice_len: int | None = 4096,
        standardize_trace: bool = True,
        return_eq_code: bool = False,
        slice_mode: str = "random",
        return_station_id: bool = False,
        events_csv: Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.traces_path = self.data_root / "z_traces.npy"
        if not self.traces_path.exists():
            raise FileNotFoundError(self.traces_path)

        self.eq_codes_path = self.data_root / "z_eq_codes.csv"
        self.conditions_path = self.data_root / "z_conditions.npy"
        if not self.conditions_path.exists():
            raise FileNotFoundError(self.conditions_path)

        self.eq_codes = None
        if self.eq_codes_path.exists():
            self.eq_codes = np.loadtxt(
                self.eq_codes_path,
                delimiter=",",
                skiprows=1,
                dtype=str,
            )

        self.condition_columns = None
        columns_path = self.data_root / "z_condition_columns.json"
        if columns_path.exists():
            self.condition_columns = json.loads(columns_path.read_text())
        else:
            self.condition_columns = [
                "lat",
                "lon",
                "depth",
                "mag",
                "distance_km",
                "az_sin",
                "az_cos",
            ]

        cond_json_candidates = [
            self.data_root / "z_condition_norm.json",
            self.data_root / "norm_stats.json",
        ]
        self.cond_stats_path = next((p for p in cond_json_candidates if p.exists()), None)
        self.conditions = np.load(self.conditions_path)  # shape [N, C]
        if self.cond_stats_path:
            self.cond_stats = json.loads(self.cond_stats_path.read_text())
        else:
            self.cond_stats = {}

        self.traces = np.load(self.traces_path, mmap_mode="r")
        if self.traces.shape[0] != self.conditions.shape[0]:
            raise ValueError(
                f"Trace count {self.traces.shape[0]} does not match "
                f"conditions {self.conditions.shape[0]}"
            )

        self.slice_len = slice_len
        self.standardize_trace = standardize_trace
        self.return_eq_code = return_eq_code
        self.return_station_id = return_station_id
        allowed_modes = {"random", "center"}
        if slice_mode not in allowed_modes:
            raise ValueError(f"slice_mode must be one of {allowed_modes}")
        self.slice_mode = slice_mode

        if self.slice_len is not None and self.slice_len > self.traces.shape[1]:
            raise ValueError(
                f"slice_len {self.slice_len} > trace length {self.traces.shape[1]}"
            )

        picks_path = self.data_root / "picks.csv"
        self.picks = None
        if picks_path.exists():
            self.picks = pd.read_csv(picks_path)

        self.station_ids: np.ndarray | None = None
        self.station_to_index: dict[str, int] = {}
        self.index_to_station: list[str] = []
        self.num_stations: int = 0

        if self.eq_codes is not None:
            event_path_candidates = []
            if events_csv is not None:
                event_path_candidates.append(Path(events_csv))
            event_path_candidates.extend(
                [
                    self.data_root / "processed_events.csv",
                    self.data_root.parent / "processed_events.csv",
                    self.data_root.parent.parent / "processed_events.csv",
                ]
            )
            events_path = next((p for p in event_path_candidates if p and p.exists()), None)
            if events_path is not None:
                events_df = pd.read_csv(events_path, dtype={"eq_code": str})
                if "station" in events_df.columns:
                    eq_to_station = dict(
                        zip(events_df["eq_code"].astype(str), events_df["station"].astype(str))
                    )
                    unique_stations: list[str] = []
                    for code in self.eq_codes:
                        station = eq_to_station.get(code)
                        if station is None:
                            continue
                        if station not in unique_stations:
                            unique_stations.append(station)
                    self.station_to_index = {station: idx + 1 for idx, station in enumerate(unique_stations)}
                    self.index_to_station = ["<pad>"] + unique_stations
                    station_ids = []
                    for code in self.eq_codes:
                        station = eq_to_station.get(code)
                        station_ids.append(self.station_to_index.get(station, 0))
                    self.station_ids = np.asarray(station_ids, dtype=np.int64)
                    self.num_stations = len(self.station_to_index) + 1
                else:
                    self.num_stations = 0
            else:
                self.num_stations = 0

    def __len__(self) -> int:
        return self.traces.shape[0]

    def __getitem__(self, index: int):
        trace = np.array(self.traces[index], copy=True)

        if self.slice_len is not None and self.slice_len < trace.shape[-1]:
            if self.slice_mode == "random":
                start = np.random.randint(0, trace.shape[-1] - self.slice_len + 1)
            else:  # center
                start = (trace.shape[-1] - self.slice_len) // 2
            trace = trace[start : start + self.slice_len]

        if self.standardize_trace:
            mean = float(trace.mean())
            std = float(trace.std())
            if std > 0:
                trace = (trace - mean) / std

        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)
        cond_tensor = torch.from_numpy(self.conditions[index]).float()

        outputs: list[object] = [trace_tensor, cond_tensor]

        if self.return_station_id:
            station_id = 0
            if self.station_ids is not None:
                station_id = int(self.station_ids[index])
            outputs.append(torch.tensor(station_id, dtype=torch.long))

        if self.return_eq_code and self.eq_codes is not None:
            outputs.append(self.eq_codes[index])

        if len(outputs) == 2:
            return trace_tensor, cond_tensor
        return tuple(outputs)


def build_dataloader(
    data_root: Path = DATA_ROOT,
    batch_size: int = 16,
    slice_len: int | None = 4096,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    standardize_trace: bool = True,
    return_eq_code: bool = False,
    slice_mode: str = "random",
    return_station_id: bool = False,
    events_csv: Path | None = None,
):
    """Construct a DataLoader for the Stage 1 dataset."""
    dataset = SeismicZDataset(
        data_root=data_root,
        slice_len=slice_len,
        standardize_trace=standardize_trace,
        return_eq_code=return_eq_code,
        slice_mode=slice_mode,
        return_station_id=return_station_id,
        events_csv=events_csv,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return dataset, loader


if __name__ == "__main__":
    dataset, loader = build_dataloader(batch_size=8, slice_len=4096, return_station_id=True)
    batch = next(iter(loader))
    traces, conds, stations = batch
    print("traces:", traces.shape, traces.dtype)
    print("conditions:", conds.shape, conds.dtype)
    print("stations:", stations.shape, stations.dtype)


