## KO_ARMT Seismic Diffusion Project (Transfer README - No Models)

This README captures the full working knowledge of the Stage 1 (Z-component at station `ARMT`) pipeline so you can set it up and retrain on another machine without copying model checkpoints.

### Goals
- Build a conditional diffusion model to synthesize realistic Z-component seismograms at `ARMT`, conditioned on event metadata.
- Stage 1 focuses on single-station, single-component (Z). Future stages will expand to 3C and multi-station.

### What to Transfer (Everything Except Models)
- Code: `*.py` at the project root (`inspect_stage1_data.py`, `prepare_stage1_data.py`, `prepare_stage1_v2.py`, `trim_stage1_artifacts.py`, `dataset_stage1.py`, `model_stage1.py`, `train_stage1.py`, `evaluate_stage1.py`, `ui_stage1.py`)
- Configs: `configs/` (e.g., `stage1_baseline.yaml`, `stage1_v2.yaml`, `stage1_v3.yaml`, and `configs/README.md`)
- Data: `processed_events.csv`, `timeseries_Z.csv`
- Artifacts (optional but recommended to skip preprocessing time): `artifacts/v1/`, `artifacts/v2/`, and/or `artifacts/v3_trimmed/`
- Docs: `docs/STAGE1_README.md`
- This README: `README_TRANSFER.md`
- Do NOT bring trained models: exclude `stage1_runs/` and any `*.pt`, `*.pth`, `*.ckpt`

### Directory Structure (key items)
- `processed_events.csv` — per-event metadata (`eq_code`, `time`, `mag`, `lat`, `lon`, `depth`, `station`, `distance_km`, `azimuth`, …)
- `timeseries_Z.csv` — Z-component waveforms (first 16 columns are metadata, then samples)
- `artifacts/` — preprocessed arrays for fast training/eval
  - `v1/`: baseline arrays
  - `v2/`: aligned windows, global normalization, extra conditioning (e.g., `tau_samples`, `log_rms`)
  - `v3_trimmed/`: v2 windows with first 10 s removed, adjusted `tau_samples`, refreshed normalization stats
- `configs/` — hyperparameters (dataset, model, training, losses)
- `stage1_runs/` — training outputs (checkpoints, logs). Exclude this when transferring.
- `docs/STAGE1_README.md` — Stage 1 technical documentation

### Environment Setup (new machine)
1) Create environment (Python 3.10–3.12 works):
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
2) Install PyTorch per your CUDA stack:
- CPU only (always works): `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- CUDA 12.x (for RTX 3050 typical): consult PyTorch install selector and use the matching command.

Notes:
- macOS MPS (Apple Silicon) is supported but has limited memory. Prefer CUDA on NVIDIA GPUs.
- Streamlit UI needs `streamlit` and `plotly` (included in `requirements.txt`).

### Data Preparation
You can train from the original CSVs (slower) or from pre-built artifacts (faster).

- Baseline v1 artifacts:
```
python prepare_stage1_data.py \
  --data-root . \
  --out-root artifacts/v1
```

- Strengthened v2 artifacts (aligned windows, global normalization, extra conditioning like `tau_samples`, `log_rms`):
```
python prepare_stage1_v2.py \
  --data-root . \
  --out-root artifacts/v2
```

- Trimmed v3 artifacts (remove first 10 s to suppress flat prefixes, recompute stats):
```
python trim_stage1_artifacts.py \
  --source-dir artifacts/v2 \
  --output-dir artifacts/v3_trimmed \
  --force-cut-samples 1000
```

Expected outputs:
- `artifacts/*/z_traces.npy` — memory-mapped traces
- `artifacts/*/conditions.npy` — conditioning vectors
- Normalization files (e.g., `*_norm.json`)
- Event code lists and indexing files as needed by the dataset

### Dataset and DataLoader
- `dataset_stage1.py` defines `SeismicZDataset`:
  - Loads `z_traces.npy` via `np.load(..., mmap_mode="r")`
  - Extracts fixed-length windows (`slice_len`) with `slice_mode` = `"random"` or `"center"`
  - Normalization: per-trace (v1) or shared global stats (v2/v3)
  - Condition vector typically includes: magnitude, latitude, longitude, depth, distance_km, azimuth, plus optional `tau_samples` and `log_rms`
  - Can return station indices (via `return_station_id=True`) by mapping `processed_events.csv` metadata; embedding padding index reserved for classifier-free guidance

### Model: Conditional UNet1D with Diffusion
- `model_stage1.py` defines `ConditionalUNet1D` and `UNet1DConfig`
  - 1D UNet backbone with ×16 down/upsampling, larger kernels, three residual blocks per scale
  - Dilated bottleneck with optional self-attention
  - Sinusoidal timestep embeddings + FiLM modulation of residual blocks
  - Learned station embeddings projected into the conditioning stream (padding index = 0 for CFG)
- Diffusion details:
  - Cosine noise schedule, v-prediction objective by default
  - Classifier-free guidance via joint dropout on conditioning vectors and station IDs
  - EMA weights used for sampling in both CLI and UI

### Training
- Entrypoint: `train_stage1.py` (drives DDPM/DDIM training)
- Key features:
  - Config-driven (see `configs/stage1_v3.yaml` for the current trimmed pipeline)
- CUDA-first loop with automatic device selection (fails fast if CUDA unavailable when `require_cuda: true`), gradient accumulation (`effective_batch_size`), warmup + cosine LR, EMA, CSV logging, and optional TensorBoard summaries (`tensorboard: true` → `stage1_runs/<config>/tb/`)
  - Energy-weighted time-domain loss, stronger spectral (log-magnitude) penalty, and RMS amplitude calibration
  - Station-aware classifier-free guidance (drop both conditioning vector and station embedding)
  - CSV logging of loss components + LR, on-the-fly sampling previews saved to `stage1_runs/<config>/`
- Example:
```
python train_stage1.py \
  --config configs/stage1_v3.yaml \
  --data-root . \
  --artifact-version v3_trimmed \
  --run-name stage1_v3
```
- Note: `training.require_cuda: true` in `stage1_v3.yaml` will raise if the process is not running on a CUDA device.

Performance notes (RTX 3050 ballpark):
- `slice_len≈2992`, batch size 8, effective batch 32 → ~0.05–0.12 s/step (optimizer steps after accumulation)
- Training 120 epochs × ~(len(dataset)/effective_batch) ≈ 2–3 hours including sampling/checkpoint overhead
- macOS M2 (MPS) can still overheat/OOM; reduce `batch_size` or run purely on CPU for sampling.

### Evaluation
- Entrypoint: `evaluate_stage1.py`
- Metrics per event and summary:
  - MAE, MSE
  - Max cross-correlation
  - Spectral L2 (FFT-domain)
  - Std, peak-to-peak amplitude
```
python evaluate_stage1.py \
  --artifact-version v3_trimmed \
  --samples-per-event 1 \
  --max-events 100
```
Outputs:
- `metrics_per_event.csv`, `metrics_summary.json`
- Auto-loads `config_used.yaml` to rebuild the UNet/loss stack, applies EMA weights, and uses station IDs if available.

### Streamlit UI
```
streamlit run ui_stage1.py
```
Features:
- Event selector by `eq_code`
- Shows real vs generated waveforms, basic stats in micro-units
- Dataset version selector (v1/v2/v3_trimmed)
- Sampler controls (DDPM/DDIM, guidance scale, DDIM eta)
- Automatic input length adjustment to multiples of total downsample factor (16)
- Unique `key` for all widgets to avoid `StreamlitDuplicateElementId`

### Known Issues and Fixes (highlights)
- CSV parsing: first 16 columns are metadata; slice payload accordingly
- NumPy memmap format: save arrays in standard `.npy` to enable `np.load(..., mmap_mode="r")`
- UNet shape mismatches: ensured channels and spatial sizes match across up/down paths
- Input length divisibility: enforce length % 16 == 0 for UNet (v3 uses ×16 downsample)
- MPS OOM/thermal throttling: use smaller batch/length; sample on CPU; prefer CUDA when available
- `torch.load` on different hardware: use `map_location='cpu'` when loading checkpoints (if you ever load models)

### Underfitting Diagnosis (Stage 1)
- Observations: weak replication of onset and amplitude decay; metrics show small gains
- Probable causes:
  - Limited dataset size
  - Model capacity not large enough
  - Loss shaping not strong enough (spectral/early-window weights may need tuning)
  - Conditioning strength (consider stronger CFG at sampling; ensure good normalization of conditions)
  - Windowing and alignment sensitivity (accuracy of `tau_samples`)

### Practical Tips
- Seeds: set and log seeds for reproducibility (PyTorch, NumPy, Python)
- Checkpointing: disable or reduce frequency during profiling; enable EMA for best sampling
- Monitoring: persist losses and metrics per epoch to track convergence
- Hardware: NVIDIA GPU with CUDA preferred; keep an eye on VRAM when increasing model width/depth

### Packaging Without Models
- Use the helper script `./pack_without_models.sh` (see below) or run:
```
tar -czf KO_ARMT_data_no_models.tar.gz \
  --exclude='stage1_runs' \
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' \
  --exclude='wandb' \
  .
```
Validation:
```
tar -tzf KO_ARMT_data_no_models.tar.gz | grep -E '\.(pt|pth|ckpt)$' || echo "No model files found"
```

### Roadmap (Next Stages)
- Stage 2: 3-component generation (Z/N/E), shared encoder with separate heads or multi-channel UNet; extend conditioning to include component-specific features and instrument responses
- Stage 3: Multi-station generation with shared event conditioning; spatial graph/attention across stations; station embeddings; distance/azimuth-aware conditioning
- Improved evaluation: magnitude-/distance-stratified metrics, spectral coherence, windowed correlation around predicted arrivals
- Data scaling: augmentations, more events, better alignment (robust cross-correlation), station-specific calibration

### Files of Interest
- `prepare_stage1_data.py` — v1 artifacts (CSV → `.npy` memmaps)
- `prepare_stage1_v2.py` — v2 artifacts (alignment, global normalization, extra features)
- `trim_stage1_artifacts.py` — trims low-energy prefixes and writes `artifacts/v3_trimmed/`
- `dataset_stage1.py` — PyTorch `Dataset` for Z-traces
- `model_stage1.py` — 1D Conditional UNet, cosine schedule, v-pred, CFG
- `train_stage1.py` — training loop (early-window, spectral loss, EMA, clipping)
- `evaluate_stage1.py` — metrics, aggregation, loading configs from checkpoint
- `ui_stage1.py` — Streamlit UI
- `configs/stage1_baseline.yaml`, `configs/stage1_v2.yaml`, `configs/stage1_v3.yaml` — hyperparameters and features
- `docs/STAGE1_README.md` — detailed Stage 1 engineering notes

If anything fails on the new machine, start by verifying:
1) Python and package versions (`requirements.txt`)
2) CUDA/PyTorch compatibility
3) Artifact presence (`artifacts/v3_trimmed`) or re-run prepare/trim scripts
4) Config paths and `DATA_ROOT`/CLI flags


