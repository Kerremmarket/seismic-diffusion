#!/usr/bin/env python3
"""Training script for Stage 1 conditional diffusion on KO_ARMT Z traces."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_stage1 import build_dataloader, SeismicZDataset
from model_stage1 import ConditionalUNet1D, UNet1DConfig


def select_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    schedule: str = "linear",
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)
    if schedule == "cosine":
        s = 0.008
        steps = num_timesteps
        x = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999)
    raise ValueError(f"Unknown schedule: {schedule}")


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"
    target: str = "eps"
    classifier_free_dropout: float = 0.0


class ExponentialMovingAverage:
    def __init__(self, params, decay: float):
        self.decay = decay
        self.shadow_params = [p.detach().clone() for p in params]
        self.collected_params = None

    @torch.no_grad()
    def update(self, params):
        for shadow, param in zip(self.shadow_params, params):
            shadow.sub_((1 - self.decay) * (shadow - param.detach()))

    def copy_to(self, params):
        for param, shadow in zip(params, self.shadow_params):
            if shadow.device != param.device:
                param.data.copy_(shadow.to(param.device))
            else:
                param.data.copy_(shadow.data)

    def store(self, params):
        self.collected_params = [p.detach().clone() for p in params]

    def restore(self, params):
        if self.collected_params is None:
            return
        for param, backup in zip(params, self.collected_params):
            param.data.copy_(backup.data)
        self.collected_params = None


class DiffusionTrainer:
    def __init__(
        self,
        model: ConditionalUNet1D,
        diffusion_cfg: DiffusionConfig,
        device: torch.device,
        sample_rate: float = 100.0,
        loss_cfg: Optional[dict] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.prediction_type = diffusion_cfg.target
        self.classifier_free_dropout = diffusion_cfg.classifier_free_dropout
        self.sample_rate = sample_rate
        self.loss_cfg = loss_cfg or {}
        self.spectral_weight = float(self.loss_cfg.get("spectral_weight", 0.0))
        self.energy_weight_alpha = float(self.loss_cfg.get("energy_weight_alpha", 0.0))
        energy_window_sec = float(self.loss_cfg.get("energy_window_seconds", 0.0))
        self.energy_window_samples = int(round(energy_window_sec * self.sample_rate))
        if self.energy_window_samples % 2 == 0 and self.energy_window_samples > 0:
            self.energy_window_samples += 1
        if self.energy_window_samples <= 1 or self.energy_weight_alpha <= 0.0:
            self.energy_window_samples = 0
            self.energy_weight_alpha = 0.0
        self.amp_loss_weight = float(self.loss_cfg.get("amp_loss_weight", 0.0))

        betas = make_beta_schedule(
            diffusion_cfg.num_timesteps,
            diffusion_cfg.beta_start,
            diffusion_cfg.beta_end,
            schedule=diffusion_cfg.schedule,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.num_timesteps = diffusion_cfg.num_timesteps
        self.register_buffers(betas, alphas, alphas_cumprod)

    def register_buffers(self, betas, alphas, alphas_cumprod):
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def _compute_energy_weights(self, traces: torch.Tensor) -> torch.Tensor | None:
        if self.energy_window_samples <= 1 or self.energy_weight_alpha <= 0.0:
            return None
        padding = self.energy_window_samples // 2
        padded = F.pad(traces ** 2, (padding, padding), mode="reflect")
        envelope = torch.sqrt(
            F.avg_pool1d(padded, kernel_size=self.energy_window_samples, stride=1)
        )
        max_env = envelope.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        weights = (envelope / max_env).pow(self.energy_weight_alpha)
        return weights.unsqueeze(1)

    def training_step(self, batch):
        if len(batch) == 4:
            traces, conditions, station_ids, _ = batch
        elif len(batch) == 3 and torch.is_tensor(batch[2]):
            traces, conditions, station_ids = batch
        else:
            traces, conditions = batch[:2]
            station_ids = None

        traces = traces.to(self.device)
        conditions = conditions.to(self.device)
        station_ids_tensor = station_ids.to(self.device) if station_ids is not None else None

        bsz = traces.size(0)
        t = torch.randint(0, self.num_timesteps, (bsz,), device=self.device)
        noise = torch.randn_like(traces)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        noisy_traces = sqrt_alpha_bar * traces + sqrt_one_minus_alpha_bar * noise

        if self.classifier_free_dropout > 0.0:
            mask = torch.rand(bsz, device=self.device) < self.classifier_free_dropout
            if mask.any():
                conditions = conditions.clone()
                conditions[mask] = 0.0
                if station_ids_tensor is not None:
                    station_ids_tensor = station_ids_tensor.clone()
                    station_ids_tensor[mask] = 0

        model_out = self.model(noisy_traces, t.float(), conditions, station_ids_tensor)

        if self.prediction_type == "v":
            target = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * traces
            pred_noise = sqrt_one_minus_alpha_bar * model_out + sqrt_alpha_bar * traces
        else:
            target = noise
            pred_noise = model_out

        error = model_out - target
        weights = self._compute_energy_weights(traces)
        if weights is not None:
            time_loss = (weights * error**2).mean()
        else:
            time_loss = F.mse_loss(model_out, target)

        spectral_loss = torch.zeros(1, device=self.device)
        if self.spectral_weight > 0.0:
            x0_hat = (noisy_traces - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
            real_fft = torch.fft.rfft(traces.squeeze(1), norm="ortho")
            synth_fft = torch.fft.rfft(x0_hat.squeeze(1), norm="ortho")
            real_mag = torch.log1p(real_fft.abs())
            synth_mag = torch.log1p(synth_fft.abs())
            spectral_loss = F.l1_loss(synth_mag, real_mag)

        amp_loss = torch.zeros(1, device=self.device)
        if self.amp_loss_weight > 0.0:
            x0_hat = (noisy_traces - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
            rms_pred = torch.sqrt(torch.clamp(x0_hat.pow(2).mean(dim=-1), min=1e-8))
            rms_true = torch.sqrt(torch.clamp(traces.pow(2).mean(dim=-1), min=1e-8))
            amp_loss = F.mse_loss(torch.log(rms_pred + 1e-6), torch.log(rms_true + 1e-6))

        total_loss = time_loss
        if self.spectral_weight > 0.0:
            total_loss = total_loss + self.spectral_weight * spectral_loss
        if self.amp_loss_weight > 0.0:
            total_loss = total_loss + self.amp_loss_weight * amp_loss

        return {
            "total_loss": total_loss,
            "time_loss": time_loss,
            "spectral_loss": spectral_loss,
            "amp_loss": amp_loss,
        }

    @torch.no_grad()
    def sample_traces(
        self,
        cond: torch.Tensor,
        slice_len: int,
        steps: Optional[int] = None,
        sample_device: Optional[torch.device] = None,
        guidance_scale: float = 1.0,
        method: str = "ddpm",
        eta: float = 0.0,
        station_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if steps is None:
            steps = self.num_timesteps
        target_device = sample_device or self.device

        betas = self.betas.to(target_device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(target_device)
        sqrt_recip_alpha = self.sqrt_recip_alphas.to(target_device)
        posterior_variance = self.posterior_variance.to(target_device)
        sqrt_alpha_bar_all = self.sqrt_alphas_cumprod.to(target_device)

        model = self.model
        was_training = model.training
        model.eval()

        current_device = next(model.parameters()).device
        moved_model = False
        if current_device != target_device:
            model.to(target_device)
            moved_model = True

        cond = cond.to(target_device)
        batch = cond.size(0)
        x = torch.randn(batch, model.cfg.in_channels, slice_len, device=target_device)
        cond_null = torch.zeros_like(cond)
        station_ids = station_ids.to(target_device) if station_ids is not None else None
        station_null = torch.zeros_like(station_ids) if station_ids is not None else None

        for t in reversed(range(steps)):
            t_batch = torch.full((batch,), t, device=target_device, dtype=torch.long)
            betas_t = betas[t_batch].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha[t_batch].view(-1, 1, 1)
            sqrt_recip_alpha_t = sqrt_recip_alpha[t_batch].view(-1, 1, 1)
            sqrt_alpha_bar_t = sqrt_alpha_bar_all[t_batch].view(-1, 1, 1)

            if guidance_scale != 1.0:
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t_batch, t_batch], dim=0)
                cond_in = torch.cat([cond, cond_null], dim=0)
                if station_ids is not None:
                    station_in = torch.cat([station_ids, station_null], dim=0)
                else:
                    station_in = None
                model_out = model(x_in, t_in.float(), cond_in, station_in)
                model_out_cond, model_out_uncond = model_out.chunk(2)
                model_out = model_out_uncond + guidance_scale * (model_out_cond - model_out_uncond)
            else:
                model_out = model(x, t_batch.float(), cond, station_ids)

            if self.prediction_type == "v":
                noise_pred = sqrt_one_minus_alpha_bar * x + sqrt_alpha_bar_t * model_out
            else:
                noise_pred = model_out

            model_mean = sqrt_recip_alpha_t * (x - betas_t / sqrt_one_minus_alpha_bar * noise_pred)
            if method == "ddim":
                alpha_bar = self.alphas_cumprod[t_batch].view(-1, 1, 1)
                alpha_bar_prev = self.alphas_cumprod_prev[t_batch].view(-1, 1, 1)
                sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                )
                pred_x0 = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
                dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.0)) * noise_pred
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
                if t > 0 and eta > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                if t > 0:
                    posterior_var = posterior_variance[t_batch].view(-1, 1, 1)
                    noise = torch.randn_like(x)
                    x = model_mean + torch.sqrt(posterior_var) * noise
                else:
                    x = model_mean

        if moved_model:
            model.to(current_device)
            x = x.to(current_device)
        if was_training:
            model.train()
        return x


def save_checkpoint(
    path: Path,
    model: ConditionalUNet1D,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
    ema: Optional[ExponentialMovingAverage],
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if ema is not None:
        payload["ema"] = [p.cpu() for p in ema.shadow_params]
    torch.save(payload, path)
    print(f"[checkpoint] saved to {path}")


def load_checkpoint(
    path: Path,
    model: ConditionalUNet1D,
    optimizer: Optional[optim.Optimizer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and "ema" in payload:
        for shadow_param, saved in zip(ema.shadow_params, payload["ema"]):
            shadow_param.copy_(saved)
    epoch = payload.get("epoch", 0)
    global_step = payload.get("global_step", 0)
    print(f"[checkpoint] loaded from {path}")
    return epoch, global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 diffusion training")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file describing dataset/training setup")
    parser.add_argument("--data-root", type=Path, default=Path("/Users/keremciftci/Desktop/KO_ARMT_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./stage1_runs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--slice-len", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Explicit device id (cpu/cuda/mps)")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--num-sample-traces", type=int, default=8)
    parser.add_argument("--sample-device", type=str, default=None, help="Device for sampling (default: training device)")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--use-ema", action="store_true", default=False)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = {}
    if args.config:
        config = yaml.safe_load(args.config.read_text()) or {}

    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    model_cfg_dict = config.get("model", {})
    diffusion_cfg_dict = config.get("diffusion", {})
    loss_cfg = config.get("loss", {})
    sampling_cfg = config.get("sampling", {})

    version = dataset_cfg.get("version", "v1")
    dataset_root = args.data_root if version == "v1" else args.data_root / "artifacts" / version
    norm_mode = dataset_cfg.get("norm_mode", "per_trace")
    standardize_trace = norm_mode == "per_trace"

    window_cfg = dataset_cfg.get("window", {})
    align_on_pick = window_cfg.get("align_on_pick", False)
    slice_len_cfg = window_cfg.get("slice_len_samples")
    if slice_len_cfg is not None:
        slice_len = int(slice_len_cfg)
    else:
        slice_len = None if align_on_pick else args.slice_len
    slice_mode = "center" if align_on_pick else "random"

    sample_rate = dataset_cfg.get("sample_rate", 100.0)

    batch_size = training_cfg.get("batch_size", args.batch_size)
    epochs = training_cfg.get("epochs", args.epochs)
    optimizer_cfg = training_cfg.get("optimizer", {})
    lr = optimizer_cfg.get("lr", args.lr)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    ema_decay = training_cfg.get("ema_decay", args.ema_decay)
    use_ema = training_cfg.get("use_ema", True if args.config else args.use_ema)
    log_every = training_cfg.get("log_every", args.log_every)
    sample_every = training_cfg.get("sample_every", args.sample_every)
    checkpoint_every = training_cfg.get("checkpoint_every", args.checkpoint_every)
    warmup_steps = training_cfg.get("warmup_steps", 0)
    effective_batch_size = training_cfg.get("effective_batch_size", batch_size)
    num_workers = training_cfg.get("num_workers", args.num_workers)
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

    grad_clip = training_cfg.get("grad_clip", args.grad_clip)

    diff_cfg = DiffusionConfig(
        num_timesteps=diffusion_cfg_dict.get("timesteps", args.timesteps),
        beta_start=diffusion_cfg_dict.get("beta_start", args.beta_start),
        beta_end=diffusion_cfg_dict.get("beta_end", args.beta_end),
        schedule=diffusion_cfg_dict.get("schedule", "linear"),
        target=diffusion_cfg_dict.get("target", "eps"),
        classifier_free_dropout=diffusion_cfg_dict.get("classifier_free_dropout", 0.0),
    )

    sample_device = torch.device(args.sample_device) if args.sample_device else None
    sample_steps = sampling_cfg.get("default_steps", args.sample_steps)
    guidance_scale = sampling_cfg.get("guidance_scale", 1.0)
    sampling_method = sampling_cfg.get("method", "ddpm")
    ddim_eta = sampling_cfg.get("ddim_eta", 0.0)
    num_sample_traces = sampling_cfg.get("preview_traces", args.num_sample_traces)

    device = select_device(args.device)
    require_cuda = training_cfg.get("require_cuda", False) or args.require_cuda
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA device required but not available. Install a GPU-enabled PyTorch build or remove require_cuda.")

    if sample_device is None:
        sample_device = device
    print(f"Using device: {device} (sampling on {sample_device})")

    run_root = args.output_dir
    if args.config:
        run_root = run_root / args.config.stem
    run_root.mkdir(parents=True, exist_ok=True)
    if config:
        (run_root / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    events_cfg_path = dataset_cfg.get("events_csv")
    events_csv = None
    if events_cfg_path:
        events_csv_candidate = Path(events_cfg_path)
        if not events_csv_candidate.is_absolute():
            events_csv_candidate = args.data_root / events_cfg_path
        if events_csv_candidate.exists():
            events_csv = events_csv_candidate
    if events_csv is None:
        candidate_list = [
            dataset_root / "processed_events.csv",
            args.data_root / "processed_events.csv",
            args.data_root.parent / "processed_events.csv" if args.data_root.parent != args.data_root else None,
        ]
        events_csv = next((p for p in candidate_list if p is not None and p.exists()), None)

    dataset, dataloader = build_dataloader(
        data_root=dataset_root,
        batch_size=batch_size,
        slice_len=slice_len,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        standardize_trace=standardize_trace,
        slice_mode=slice_mode,
        return_station_id=True,
        events_csv=events_csv,
    )
    effective_slice_len = dataset.traces.shape[1] if slice_len is None else slice_len

    cfg_model = UNet1DConfig(
        in_channels=1,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        num_res_blocks=num_res_blocks,
        kernel_size=kernel_size,
        time_embedding_dim=time_embedding_dim,
        cond_dim=dataset.conditions.shape[1],
        station_embedding_dim=station_embedding_dim,
        num_stations=getattr(dataset, "num_stations", 0),
        use_attention=use_attention,
        attention_heads=attention_heads,
        mid_dilations=mid_dilations,
    )
    model = ConditionalUNet1D(cfg_model)

    trainer = DiffusionTrainer(
        model,
        diff_cfg,
        device,
        sample_rate=sample_rate,
        loss_cfg=loss_cfg,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    grad_accum_steps = max(1, math.ceil(effective_batch_size / batch_size))
    steps_per_epoch = max(1, math.ceil(len(dataloader) / grad_accum_steps))
    total_train_steps = max(1, steps_per_epoch * epochs)

    def lr_schedule(step: int) -> float:
        if total_train_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_train_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    ema = ExponentialMovingAverage(model.parameters(), ema_decay) if use_ema else None

    start_epoch = 0
    global_step = 0
    if args.resume is not None and args.resume.exists():
        start_epoch, global_step = load_checkpoint(args.resume, model, optimizer, ema)
        start_epoch += 1

    loss_log_path = run_root / "training_log.csv"
    if not loss_log_path.exists():
        loss_log_path.write_text("step,epoch,loss,time_loss,spectral_loss,amp_loss,lr,grad_norm\n")

    use_tensorboard = training_cfg.get("tensorboard", False)
    writer: SummaryWriter | None = None
    if use_tensorboard:
        writer = SummaryWriter(log_dir=str(run_root / "tb"))

    model.train()
    optimizer.zero_grad()
    accum_steps = 0

    for epoch in range(start_epoch, epochs):
        for batch in dataloader:
            losses = trainer.training_step(batch)
            total_loss = losses["total_loss"]
            (total_loss / grad_accum_steps).backward()
            accum_steps += 1

            if accum_steps % grad_accum_steps != 0:
                continue

            grad_norm_val = None
            if grad_clip:
                grad_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if ema is not None:
                ema.update(model.parameters())

            global_step += 1
            accum_steps = 0

            if global_step % log_every == 0:
                time_val = float(losses["time_loss"].detach().item())
                spec_val = float(losses["spectral_loss"].detach().item())
                amp_val = float(losses["amp_loss"].detach().item())
                total_val = float(losses["total_loss"].detach().item())
                lr_val = optimizer.param_groups[0]["lr"]
                grad_display = grad_norm_val if grad_norm_val is not None else float("nan")
                print(
                    f"[epoch {epoch} step {global_step}] "
                    f"loss={total_val:.6f} time={time_val:.6f} "
                    f"spectral={spec_val:.6f} amp={amp_val:.6f} lr={lr_val:.6e} "
                    f"grad={grad_display:.4f}"
                )
                with loss_log_path.open("a") as log_file:
                    log_file.write(
                        f"{global_step},{epoch},{total_val:.6f},{time_val:.6f},"
                        f"{spec_val:.6f},{amp_val:.6f},{lr_val:.8f},{grad_display:.6f}\n"
                    )
                if writer is not None:
                    writer.add_scalar("loss/total", total_val, global_step)
                    writer.add_scalar("loss/time", time_val, global_step)
                    writer.add_scalar("loss/spectral", spec_val, global_step)
                    writer.add_scalar("loss/amplitude", amp_val, global_step)
                    writer.add_scalar("lr", lr_val, global_step)
                    writer.add_scalar("grad_norm", grad_display, global_step)

            if global_step % sample_every == 0:
                sample_path = run_root / f"samples_step{global_step}.npy"
                if ema is not None:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                cond_np = dataset.conditions[:num_sample_traces]
                cond = torch.from_numpy(cond_np).to(device)
                station_ids = None
                if getattr(dataset, "station_ids", None) is not None:
                    station_ids = torch.from_numpy(dataset.station_ids[:num_sample_traces]).long().to(device)
                samples = trainer.sample_traces(
                    cond,
                    effective_slice_len,
                    steps=sample_steps,
                    sample_device=sample_device,
                    guidance_scale=guidance_scale,
                    method=sampling_method,
                    eta=ddim_eta,
                    station_ids=station_ids,
                ).cpu().numpy()
                np.save(sample_path, samples)
                print(f"[sample] saved {sample_path}")
                if ema is not None:
                    ema.restore(model.parameters())

            if global_step % checkpoint_every == 0:
                ckpt_path = run_root / f"checkpoint_step{global_step}.pt"
                save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, ema)

        ckpt_path = run_root / f"checkpoint_epoch{epoch}.pt"
        save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, ema)

    if writer is not None:
        writer.close()

    print("Training complete.")


if __name__ == "__main__":
    main()


