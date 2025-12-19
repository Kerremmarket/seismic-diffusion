## Seismic Diffusion (Stage 1 – ARMT)

Conditional diffusion model that synthesizes Z‑component seismograms at station `ARMT` from event metadata. This repo bundles preprocessing scripts, training and evaluation code, a Streamlit UI, and prebuilt artifacts (aligned `v2` and trimmed `v3_trimmed`). No model checkpoints are included.

### What’s inside
- Data: `processed_events.csv`, `timeseries_Z.csv`
- Artifacts: `artifacts/v2/` (aligned + global norm) and `artifacts/v3_trimmed/` (10 s trimmed, refreshed stats)
- Code: `prepare_stage1_v2.py`, `trim_stage1_artifacts.py`, `dataset_stage1.py`, `model_stage1.py`, `train_stage1.py`, `evaluate_stage1.py`, `ui_stage1.py`
- Configs & docs: `configs/*.yaml`, `configs/README.md`, `docs/STAGE1_README.md`, `README_TRANSFER.md`

### Setup
1) Python 3.10–3.12 recommended.  
2) Install deps:
```
pip install --upgrade pip
pip install -r requirements.txt
```
3) Install PyTorch for your hardware (examples):  
   - CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`  
   - CUDA: use the command from the PyTorch selector for your CUDA version.

### Quickstart
- **Use existing artifacts** (already in `artifacts/v2` and `artifacts/v3_trimmed`). To regenerate:
```
python prepare_stage1_v2.py --data-root . --out-root artifacts/v2
python trim_stage1_artifacts.py --source-dir artifacts/v2 --output-dir artifacts/v3_trimmed --force-cut-samples 1000
```
- **Train** (trimmed pipeline):
```
python train_stage1.py --config configs/stage1_v3.yaml --artifact-version v3_trimmed --data-root . --run-name stage1_v3
```
- **Evaluate**:
```
python evaluate_stage1.py --artifact-version v3_trimmed --checkpoint <path_to_checkpoint>
```
- **UI**:
```
streamlit run ui_stage1.py
```
- **Pack without models** (optional, bash):
```
./pack_without_models.sh
```

### Directory overview
- `artifacts/` — v2 aligned artifacts and v3 trimmed bundle
- `configs/` — YAML configs + notes
- `docs/` — Stage 1 engineering notes
- `*.py` — data prep, dataset, model, training, evaluation, UI
- `png/` — waveform previews

### Notes
- Repo size ~0.5 GB due to artifacts and previews; no model checkpoints included (`stage1_runs/` is intentionally absent).
- See `docs/STAGE1_README.md` for deeper engineering details.

