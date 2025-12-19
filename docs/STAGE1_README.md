# Stage 1 Seismic Diffusion – Engineering Notes

## Overview
- Goal: conditional diffusion to synthesize ARMT Z-component seismograms given event metadata.
- Raw data: `timeseries_Z.csv`, `processed_events.csv` under repo root.
- Artifacts:
  - `artifacts/v1/` (legacy) – per-trace z-score, full 7001-sample window.
  - `artifacts/v2/` – travel-time aligned window (~40 s), global normalization, tau\_p + log RMS features.
  - `artifacts/v3_trimmed/` – v2 windows with the first 10 s removed, tau offsets adjusted, and a fresh global normalization.
- Training runs live under `stage1_runs/<config_stem>/`; evaluations under `stage1_eval/<artifact_version>/`.

## Data Pipeline
1. **Baseline (v1)**  
   - Script: `prepare_stage1_data.py` (legacy).  
   - Output files in repo root: `z_traces.npy`, `z_conditions.npy`, `z_eq_codes.csv`, `z_condition_norm.json`.
   - Conditioning vector: `[lat, lon, depth, mag, distance_km, az_sin, az_cos]`.  
   - Notes: per-trace standardization erases magnitude/distance scaling and arrivals are not aligned.

2. **Aligned pipeline (v2)**  
   - Script: `prepare_stage1_v2.py --config configs/stage1_v2.yaml`.  
   - Window aligned using travel-time tau\_p (distance / vp\_km\_per\_s); crop `[pre_seconds, length_seconds]`.  
   - Artifacts written to `artifacts/v2/`:  
     - `z_traces.npy` (global normalized float32 data)  
     - `z_conditions.npy` (+ columns JSON) containing `[lat, lon, depth, mag, distance_km, tau_samples, log_rms, az_sin, az_cos]`  
     - `z_eq_codes.csv`, `norm_stats.json`, `picks.csv` (start, length, tau method)
   - Config toggles (see `configs/stage1_v2.yaml`): normalization mode, pick method (`travel_time` vs future `sta_lta`), window length, vp, etc.  
   - To add future variants, create new YAML referencing a new output version (`artifacts/v3`, etc.) without touching existing artifacts.

3. **Trimmed prefix (v3\_trimmed)**  
   - Start from v2 artifacts and run `python trim_stage1_artifacts.py --force-cut-samples 1000` to remove the first 10 s (1000 samples).  
   - Updates `tau_samples`, recomputes global mean/std, and writes `trim_stats.json` describing the cut.  
   - Output bundle lives in `artifacts/v3_trimmed/` with refreshed conditions, picks, and normalization metadata.  
   - Training config: `configs/stage1_v3.yaml` sets `slice_len_samples=2992` (divisible by the ×16 UNet stride) and points the trainer at the trimmed dataset.

4. **Configs**  
   - `configs/stage1_baseline.yaml`: reproduces v1 settings.  
   - `configs/stage1_v2.yaml`: upgraded alignment/global norm/cosine schedule/v-pred.  
   - `configs/stage1_v3.yaml`: trimmed data, higher-capacity UNet, energy-weighted losses, cosine LR with warmup.  
   - Each config has `dataset`, `training`, `diffusion`, `loss`, `sampling` sections.  
   - Use `prepare_stage1_v2.py --config ...` followed by `train_stage1.py --config ...`.

## Model & Training
- UNet (`model_stage1.py`):
  - Base width 128 with channel multipliers `[1, 2, 4, 8]` (×16 downsample factor).
  - Three residual blocks per scale, larger kernels, dilated bottleneck, and optional self-attention.
  - FiLM conditioning augmented with learned station embeddings (padding index used for classifier-free guidance).
- `train_stage1.py`:
  - YAML-driven: dataset selection, normalization mode, sample rate, window length.
  - Diffusion settings: cosine noise schedule, v-prediction, classifier-free dropout.
  - Loss stack: energy-weighted time-domain MSE, stronger spectral L1, RMS amplitude calibration.
  - CUDA-first loop with gradient accumulation (`effective_batch_size`), warmup + cosine LR (`LambdaLR`), EMA, CSV logging, and optional TensorBoard summaries (`training.tensorboard: true` writes to `stage1_runs/<config>/tb/`).
  - Preview sampling honors EMA weights and writes to `stage1_runs/<config>/samples_step*.npy`.
- Training command examples:  
  ```bash
  python train_stage1.py --config configs/stage1_v3.yaml --device cuda
  ```

## Evaluation
- `evaluate_stage1.py --artifact-version v3_trimmed --checkpoint <path>` computes metrics on a time-based holdout split:  
  - MAE/MSE (raw & lag-aligned), max corr, spectral L2, std/ptp.  
  - Automatically reloads `config_used.yaml` to reconstruct the upgraded UNet and loss stack, applies EMA weights, and keeps station embeddings in sync.  
  - Saves per-event CSV + summary JSON in `stage1_eval/<version>/`.  
  - Extendable via additional metrics (e.g., PSD band errors, envelope MAE).

## Streamlit UI (`ui_stage1.py`)
- Features:
  - Dataset version selector (`v1`, `v2`, …).  
  - Event metadata + conditioning feature table + alignment info (τ\_p, window).  
  - Generation controls: checkpoint path, diffusion steps, sampler (DDPM/DDIM), guidance scale, seed.  
  - Overlays with alignment/normalization toggles and stats.  
  - Evaluation tab reads `stage1_eval/<version>/` outputs and shows per-event metrics.  
- Launch:  
  ```bash
  streamlit run ui_stage1.py
  ```

## Workflow
1. **Prepare data**: run `prepare_stage1_v2.py` (or future configs).  
2. **Train**: `train_stage1.py --config …`, monitor logs; checkpoints & samples in `stage1_runs/<config>/`.  
3. **Evaluate**: `python evaluate_stage1.py --artifact-version <version> --checkpoint <path>`.  
4. **Review**: open UI, pick same artifact version, load checkpoint (auto-reads `config_used.yaml`), inspect overlays + metrics.

## Future TODO
- Implement STA/LTA pick module once quality picks available.  
- Multi-resolution/STFT loss integration.  
- Extend to NS/EW (3-component) and multi-station conditioning.  
- Automate experiment registry (sweeps, W&B logging).  
- Additional evaluation panels (PSD comparison, envelope plots) in the UI.

