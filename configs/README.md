# Stage 1 Configs

- `stage1_baseline.yaml`: reproduces the current v1 pipeline (per-trace normalization, full window).
- `stage1_v2.yaml`: travel-time alignment, tau_p feature, global normalization, cosine/v-pred, classifier-free guidance, and DDIM sampling.
- `stage1_v3.yaml`: trimmed-prefix dataset, higher-capacity UNet (×16 downsample, attention, dilations), energy-weighted losses, gradient accumulation, cosine LR with warmup.

Each config has three sections:
- `dataset`: controls preprocessing artifacts (version, normalization, pick method, window).
- `training` & `diffusion`: UNet + diffusion hyperparameters.
- `loss` and `sampling`: loss weighting and default sampler options.

Use `python prepare_stage1_v2.py --config configs/stage1_v2.yaml` to generate new artifacts, and `python train_stage1.py --config configs/stage1_v2.yaml` to train with that setup.

