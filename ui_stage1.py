#!/usr/bin/env python3
from __future__ import annotations

"""Streamlit UI to explore KO_ARMT Stage 1 seismic diffusion results."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

from model_stage1 import ConditionalUNet1D, UNet1DConfig
from train_stage1 import DiffusionConfig, DiffusionTrainer, ExponentialMovingAverage, load_checkpoint


def read_checkpoint_config(checkpoint_path: Path) -> dict:
    cfg: dict = {}
    if checkpoint_path is None or not checkpoint_path.exists():
        return cfg
    candidate = checkpoint_path.parent / "config_used.yaml"
    if not candidate.exists():
        parent = checkpoint_path.parent.parent
        if parent != checkpoint_path.parent:
            alt = parent / "config_used.yaml"
            if alt.exists():
                candidate = alt
    if candidate.exists():
        cfg = yaml.safe_load(candidate.read_text()) or {}
    return cfg


DATA_ROOT = Path("/Users/keremciftci/Desktop/KO_ARMT_data")
ARTIFACT_ROOT = DATA_ROOT / "artifacts"
SAMPLING_RATE = 100.0  # Hz
DEFAULT_DIFFUSION_TIMESTEPS = 200


@st.cache_resource
def available_versions() -> list[str]:
    versions = ["v1"]
    if ARTIFACT_ROOT.exists():
        for sub in sorted(ARTIFACT_ROOT.iterdir()):
            if (sub / "z_traces.npy").exists():
                versions.append(sub.name)
    return versions


@st.cache_resource
def load_dataset(version: str):
    base = DATA_ROOT if version == "v1" else ARTIFACT_ROOT / version
    traces = np.load(base / "z_traces.npy", mmap_mode="r")
    eq_codes_path = base / "z_eq_codes.csv"
    if not eq_codes_path.exists():
        eq_codes_path = DATA_ROOT / "z_eq_codes.csv"
    eq_codes_df = pd.read_csv(eq_codes_path, dtype={"eq_code": str})
    eq_codes = eq_codes_df["eq_code"].tolist()
    eq_to_index = {code: idx for idx, code in enumerate(eq_codes)}
    conditions = np.load(base / "z_conditions.npy")
    columns_path = base / "z_condition_columns.json"
    if columns_path.exists():
        condition_columns = json.loads(columns_path.read_text())
    else:
        condition_columns = ["lat", "lon", "depth", "mag", "distance_km", "az_sin", "az_cos"]
    norm_candidates = [
        base / "norm_stats.json",
        base / "z_condition_norm.json",
    ]
    norm_stats = {}
    for path in norm_candidates:
        if path.exists():
            norm_stats = json.loads(path.read_text())
            break
    picks = None
    picks_path = base / "picks.csv"
    if picks_path.exists():
        picks = pd.read_csv(picks_path, dtype={"eq_code": str})

    station_ids = None
    station_vocab: dict[str, int] = {}
    events_candidates = [
        base / "processed_events.csv",
        DATA_ROOT / "processed_events.csv",
    ]
    events_path = next((p for p in events_candidates if p.exists()), None)
    if events_path is not None:
        events_df = pd.read_csv(events_path, dtype={"eq_code": str, "station": str})
        eq_to_station = dict(zip(events_df["eq_code"], events_df["station"]))
        ids: list[int] = []
        for code in eq_codes:
            station = eq_to_station.get(code)
            if station is None:
                ids.append(0)
                continue
            if station not in station_vocab:
                station_vocab[station] = len(station_vocab) + 1
            ids.append(station_vocab.get(station, 0))
        station_ids = np.asarray(ids, dtype=np.int64)

    return {
        "base": base,
        "traces": traces,
        "eq_codes": eq_codes,
        "eq_to_index": eq_to_index,
        "conditions": conditions,
        "condition_columns": condition_columns,
        "norm_stats": norm_stats,
        "picks": picks,
        "station_ids": station_ids,
        "station_vocab": station_vocab,
    }


@st.cache_resource
def load_events():
    df = pd.read_csv(DATA_ROOT / "processed_events.csv", dtype={"eq_code": str})
    return df.set_index("eq_code")


def find_latest_checkpoint(default_dir: Path, subdir: Optional[str] = None) -> Optional[Path]:
    search_dir = default_dir / subdir if subdir else default_dir
    if not search_dir.exists():
        return None
    candidates = list(search_dir.glob("checkpoint_*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@st.cache_resource
def load_sampler(
    checkpoint_path: str,
    cond_dim: int,
    num_stations: int,
) -> tuple[DiffusionTrainer, dict]:
    ckpt = Path(checkpoint_path)
    device = torch.device("cpu")

    config = read_checkpoint_config(ckpt)

    training_cfg = config.get("training", {})
    model_cfg_dict = config.get("model", {})
    base_channels = model_cfg_dict.get("base_channels", training_cfg.get("base_channels", 64))
    channel_multipliers = tuple(
        model_cfg_dict.get("channel_multipliers", training_cfg.get("channel_multipliers", [1, 2, 4]))
    )
    num_res_blocks = model_cfg_dict.get("num_res_blocks", training_cfg.get("num_res_blocks", 2))
    kernel_size = model_cfg_dict.get("kernel_size", training_cfg.get("kernel_size", 3))
    time_embedding_dim = model_cfg_dict.get("time_embedding_dim", 256)
    station_embedding_dim = model_cfg_dict.get("station_embedding_dim", 0)
    use_attention = model_cfg_dict.get("use_attention", True)
    attention_heads = model_cfg_dict.get("attention_heads", 4)
    mid_dilations = tuple(model_cfg_dict.get("mid_dilations", (1, 2, 4)))

    cfg = UNet1DConfig(
        in_channels=1,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        num_res_blocks=num_res_blocks,
        kernel_size=kernel_size,
        time_embedding_dim=time_embedding_dim,
        cond_dim=cond_dim,
        station_embedding_dim=station_embedding_dim,
        num_stations=num_stations,
        use_attention=use_attention,
        attention_heads=attention_heads,
        mid_dilations=mid_dilations,
    )

    diffusion_cfg = config.get("diffusion", {})
    diff_cfg = DiffusionConfig(
        num_timesteps=diffusion_cfg.get("timesteps", DEFAULT_DIFFUSION_TIMESTEPS),
        beta_start=diffusion_cfg.get("beta_start", 1e-4),
        beta_end=diffusion_cfg.get("beta_end", 0.02),
        schedule=diffusion_cfg.get("schedule", "linear"),
        target=diffusion_cfg.get("target", "eps"),
        classifier_free_dropout=diffusion_cfg.get("classifier_free_dropout", 0.0),
    )

    dataset_cfg = config.get("dataset", {})
    sample_rate = dataset_cfg.get("sample_rate", SAMPLING_RATE)

    model = ConditionalUNet1D(cfg)
    trainer = DiffusionTrainer(model, diff_cfg, device, sample_rate=sample_rate, loss_cfg=config.get("loss", {}))

    ema = ExponentialMovingAverage(model.parameters(), decay=0.0)
    load_checkpoint(ckpt, model, optimizer=None, ema=ema)
    ema.copy_to(model.parameters())
    model.eval()
    return trainer, config


def plot_waveform(time_axis, data, label, color, fig=None):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=data, name=label, line=dict(color=color)))
    fig.update_layout(
        margin=dict(l=30, r=30, t=30, b=30),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
    )
    return fig


def zscore(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if std > 0:
        return (arr - mean) / std
    return arr * 0.0


def max_corr_lag(reference: np.ndarray, candidate: np.ndarray) -> int:
    """Return lag (in samples) that maximizes cross-correlation between two equal-length vectors."""
    corr = np.correlate(reference, candidate, mode="full")
    lag = int(np.argmax(corr) - (len(reference) - 1))
    return lag


def shift_with_lag(array: np.ndarray, lag: int, target_len: int) -> np.ndarray:
    """Shift array by lag samples (positive lag shifts right) and trim/pad to target_len."""
    if lag == 0:
        return array[:target_len]
    if lag > 0:
        shifted = np.concatenate([np.zeros(lag, dtype=np.float32), array])[:target_len]
    else:
        shifted = array[-lag:]
        if shifted.size < target_len:
            shifted = np.concatenate(
                [shifted, np.zeros(target_len - shifted.size, dtype=np.float32)]
            )
    return shifted.astype(np.float32)


@st.cache_data
def load_evaluation_results(version: str):
    summary_path = Path("stage1_eval") / version / "metrics_summary.json"
    per_event_path = Path("stage1_eval") / version / "metrics_per_event.csv"
    if not summary_path.exists() or not per_event_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    summary_df = pd.DataFrame(summary).T
    if {"mean", "std", "median"}.issubset(summary_df.columns):
        summary_df = summary_df[["mean", "std", "median"]]
    per_event_df = pd.read_csv(per_event_path)
    timestamp = summary_path.stat().st_mtime
    return summary_df, per_event_df, timestamp


def main():
    st.set_page_config(page_title="KO_ARMT Stage 1 Viewer", layout="wide")
    st.title("KO_ARMT Stage 1 – Z-component Diffusion Viewer")

    versions = available_versions()
    default_index = versions.index("v2") if "v2" in versions else 0
    with st.sidebar:
        dataset_version = st.selectbox("Dataset version", versions, index=default_index)

    dataset_info = load_dataset(dataset_version)
    traces = dataset_info["traces"]
    eq_codes = dataset_info["eq_codes"]
    eq_to_index = dataset_info["eq_to_index"]
    conditions = dataset_info["conditions"]
    condition_columns = dataset_info["condition_columns"]
    picks_df = dataset_info["picks"]
    station_ids = dataset_info.get("station_ids")
    station_vocab = dataset_info.get("station_vocab", {})
    events = load_events()

    if st.session_state.get("generated_version") != dataset_version:
        st.session_state.pop("generated", None)
        st.session_state.pop("generated_steps", None)
        st.session_state["generated_version"] = dataset_version

    col_left, col_right = st.columns([2, 1])

    with col_left:
        selected_code = st.selectbox("Select Event (eq_code)", options=eq_codes, key="event_select_main")
        idx = eq_to_index[selected_code]
        trace = np.asarray(traces[idx], dtype=np.float32)
        trace_len = int(trace.shape[0])
        model_stride = 16  # updated UNet downsamples by 2 four times
        usable_len = trace_len - (trace_len % model_stride)
        if usable_len <= 0:
            usable_len = trace_len
        trace_eff = trace[:usable_len]
        time_axis = np.arange(usable_len) / SAMPLING_RATE

        st.session_state["trace_len"] = trace_len
        st.session_state["usable_len"] = usable_len

        st.write("### Real Z-component waveform")
        fig = plot_waveform(time_axis, trace_eff, label="Real", color="#1f77b4")
        st.plotly_chart(fig, width="stretch", key=f"real_waveform_{dataset_version}")

    with col_right:
        st.write("### Event metadata")
        if selected_code in events.index:
            row = events.loc[selected_code]
            meta_df = pd.DataFrame(
                {
                    "Field": ["Time", "Magnitude", "Latitude", "Longitude", "Depth (km)", "Distance (km)", "Azimuth"],
                    "Value": [
                        row["time"],
                        row["mag"],
                        row["lat"],
                        row["lon"],
                        row["depth"],
                        row["distance_km"],
                        row["azimuth"],
                    ],
                }
            )
            meta_df["Value"] = meta_df["Value"].astype(str)
            st.dataframe(meta_df, use_container_width=True)
        else:
            st.info("Metadata not found for selected event.")

        cond_series = pd.Series(conditions[idx], index=condition_columns)
        extra_cols = [c for c in cond_series.index if c not in {"lat", "lon", "depth", "mag", "distance_km"}]
        if extra_cols:
            st.write("### Condition features")
            st.dataframe(
                pd.DataFrame({"Feature": extra_cols, "Value": cond_series[extra_cols].values}),
                use_container_width=True,
            )

        if picks_df is not None:
            pick_row = picks_df.loc[picks_df["eq_code"] == selected_code]
            if not pick_row.empty:
                st.write("### Window alignment")
                pick_display = (
                    pick_row[["tau_samples", "window_start", "window_length", "tau_method"]]
                    .rename(columns={"tau_samples": "Tau (samples)", "window_start": "Window start", "window_length": "Window length", "tau_method": "Method"})
                )
                st.dataframe(pick_display, use_container_width=True)

    st.divider()
    st.write("## Generate synthetic trace")

    ckpt_default = find_latest_checkpoint(Path("stage1_runs"), subdir=dataset_version if dataset_version != "v1" else None)
    default_cfg = read_checkpoint_config(ckpt_default) if ckpt_default else {}
    sampling_defaults = default_cfg.get("sampling", {})
    default_steps = int(sampling_defaults.get("default_steps", 60 if dataset_version != "v1" else 50))
    default_method = sampling_defaults.get("method", "ddim" if dataset_version != "v1" else "ddpm")
    default_guidance = float(sampling_defaults.get("guidance_scale", 1.5 if dataset_version != "v1" else 1.0))
    default_eta = float(sampling_defaults.get("ddim_eta", 0.0))

    ckpt_path = st.text_input(
        "Checkpoint path",
        value=str(ckpt_default) if ckpt_default else "",
        help="Path to a saved checkpoint (*.pt) from train_stage1.py",
        key=f"checkpoint_input_{dataset_version}",
    )
    sample_steps = st.slider("Sampling diffusion steps", min_value=10, max_value=400, value=min(default_steps, 400), step=10)
    sampling_method = st.selectbox("Sampling method", ["ddpm", "ddim"], index=1 if default_method == "ddim" else 0)
    guidance_scale = st.slider("Guidance scale", min_value=0.0, max_value=3.0, value=default_guidance, step=0.1)
    ddim_eta = (
        st.slider("DDIM eta", min_value=0.0, max_value=1.0, value=default_eta, step=0.05)
        if sampling_method == "ddim"
        else 0.0
    )
    seed_value = st.number_input(
        "Sampling seed (makes results repeatable)",
        min_value=0,
        value=0,
        step=1,
        help="Change this to get a different random draw; same seed reproduces the same waveform.",
        key=f"seed_input_{dataset_version}",
    )

    if "generated" not in st.session_state:
        st.session_state["generated"] = None
        st.session_state["generated_steps"] = None

    generate_button = st.button("Generate synthetic trace", key=f"generate_btn_{dataset_version}")

    if generate_button:
        if not ckpt_path or not Path(ckpt_path).exists():
            st.error("Checkpoint path is invalid. Please provide a valid .pt file.")
        else:
            with st.spinner("Sampling from diffusion model..."):
                trainer, cfg_used = load_sampler(
                    checkpoint_path=ckpt_path,
                    cond_dim=conditions.shape[1],
                    num_stations=(len(station_vocab) + 1) if station_ids is not None else 0,
                )
                cfg_version = cfg_used.get("dataset", {}).get("version")
                if cfg_version and cfg_version != dataset_version:
                    st.warning(f"Checkpoint was trained on {cfg_version}; you selected {dataset_version}. Results may not match.")
                if seed_value is not None:
                    torch.manual_seed(int(seed_value))
                    np.random.seed(int(seed_value))
                cond_tensor = torch.from_numpy(conditions[idx : idx + 1]).float()
                station_tensor = None
                if station_ids is not None:
                    station_tensor = torch.tensor([int(station_ids[idx])], dtype=torch.long)
                samples = trainer.sample_traces(
                    cond_tensor,
                    slice_len=usable_len,
                    steps=sample_steps,
                    sample_device=torch.device("cpu"),
                    guidance_scale=guidance_scale,
                    method=sampling_method,
                    eta=ddim_eta if sampling_method == "ddim" else 0.0,
                    station_ids=station_tensor,
                )
                generated = samples.squeeze(0).squeeze(0).numpy()[:usable_len]
                st.session_state["generated"] = generated
                st.session_state["generated_steps"] = sample_steps

    if st.session_state.get("generated") is not None:
        generated = st.session_state["generated"]
        steps_used = st.session_state.get("generated_steps", sample_steps)
        time_axis_gen = np.arange(generated.shape[0]) / SAMPLING_RATE

        st.success(f"Generated synthetic trace (steps={steps_used}, method={sampling_method}, guidance={guidance_scale:.2f})")
        fig_gen = plot_waveform(time_axis_gen, generated, label="Synthetic", color="#d62728")
        st.plotly_chart(fig_gen, width="stretch")

        st.write("### Overlay comparison")
        align_traces = st.checkbox("Align synthetic to real via cross-correlation", value=True)
        normalize_vis = st.checkbox("Normalize traces for visualization (z-score)", value=True)

        real_arr = trace_eff.copy()
        synth_arr = np.asarray(generated, dtype=np.float32)

        min_len = min(real_arr.shape[0], synth_arr.shape[0])
        real_vis = real_arr[:min_len].copy()
        synth_vis = synth_arr[:min_len].copy()

        if normalize_vis:
            real_vis = zscore(real_vis)
            synth_vis = zscore(synth_vis)

        lag = 0
        if align_traces and min_len > 0:
            lag = max_corr_lag(real_vis, synth_vis)
            synth_vis = shift_with_lag(synth_vis, lag, min_len)

        time_overlay = np.arange(min_len) / SAMPLING_RATE
        overlay_fig = go.Figure()
        overlay_fig.add_trace(
            go.Scatter(
                x=time_overlay,
                y=real_vis,
                name="Real",
                line=dict(color="#1f77b4", width=2),
            )
        )
        overlay_fig.add_trace(
            go.Scatter(
                x=time_overlay,
                y=synth_vis,
                name="Synthetic",
                line=dict(color="#d62728", width=1.5),
                opacity=0.55,
            )
        )
        overlay_fig.update_layout(
            margin=dict(l=30, r=30, t=30, b=30),
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(overlay_fig, width="stretch")
        if align_traces and min_len > 0:
            time_shift = lag / SAMPLING_RATE
            st.caption(f"Synthetic trace shifted by {lag} samples ({time_shift:.2f} s) for overlay.")

        st.write("### Basic statistics (plotted arrays)")
        stats_df = pd.DataFrame(
            {
                "Trace": ["Real (vis)", "Synthetic (vis)"],
                "Mean": [float(real_vis.mean()), float(synth_vis.mean())],
                "Std": [float(real_vis.std()), float(synth_vis.std())],
                "Peak-to-peak": [
                    float(real_vis.max() - real_vis.min()),
                    float(synth_vis.max() - synth_vis.min()),
                ],
            }
        )
        st.dataframe(stats_df, use_container_width=True)

    st.divider()
    st.write("## Evaluation metrics")
    col_eval_left, col_eval_right = st.columns([3, 2])

    with col_eval_left:
        if st.button("Reload evaluation files", key=f"reload_eval_{dataset_version}"):
            load_evaluation_results.clear()
        eval_data = load_evaluation_results(dataset_version)
        if eval_data is None:
            st.info(
                f"Run `python evaluate_stage1.py --artifact-version {dataset_version} --checkpoint <file>` "
                "to populate evaluation metrics."
            )
        else:
            summary_df, per_event_df, timestamp = eval_data
            summary_display = summary_df.copy()
            summary_display.index.name = "Metric"
            st.subheader("Summary (latest evaluation)")
            st.dataframe(summary_display, use_container_width=True)
            st.caption(
                f"Source: stage1_eval/{dataset_version}/metrics_summary.json "
                f"(updated {pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')})"
            )

    with col_eval_right:
        eval_data = load_evaluation_results(dataset_version)
        if eval_data is not None:
            _, per_event_df, _ = eval_data
            if "eq_code" in per_event_df.columns and selected_code in per_event_df["eq_code"].values:
                event_row = per_event_df.loc[per_event_df["eq_code"] == selected_code].drop(columns=["eq_code"])
                event_row = event_row.T.reset_index()
                event_row.columns = ["Metric", "Value"]
                st.subheader("Selected event metrics")
                st.dataframe(event_row, use_container_width=True)
            else:
                st.info("Selected event not present in metrics_per_event.csv. Run evaluation to include it.")


if __name__ == "__main__":
    main()

