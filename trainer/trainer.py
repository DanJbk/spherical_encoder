import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from .config import TrainerConfig, CSVLogger

class ImageToImageTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable[[Any, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
        config: TrainerConfig,
    ):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn  # Expected to return: (total_loss, {metric_name: metric_tensor})
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = CSVLogger(self.config.output_dir / "metrics.csv")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
        # Schedulers: Linear Warmup -> Cosine Decay
        warmup = LinearLR(
            self.optimizer, start_factor=0.01, total_iters=config.warmup_epochs
        )
        cosine = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.total_epochs - config.warmup_epochs, 
            eta_min=config.min_lr
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs]
        )
        
        # Mixed Precision
        self.scaler = torch.amp.GradScaler(device=self.device.type)
        
        # Optional EMA setup
        self.ema_model = None
        if config.ema_decay is not None:
            self.ema_model = AveragedModel(
                self.model, 
                multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay)
            )
        self.current_epoch = 0
        self.best_val_loss = float('inf')

    def save_checkpoint(self, filename: str, is_best: bool = False):
        state = {
            'epoch': self.current_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'ema_state': self.ema_model.state_dict() if self.ema_model else None,
        }
        torch.save(state, self.config.output_dir / f"{filename}.pt")
        if is_best:
            torch.save(state, self.config.output_dir / "best_model.pt")

    def resume_from_checkpoint(self, checkpoint_path: str | Path):
        checkpoint_path = Path(checkpoint_path)
        
        # Edge Case 1: File doesn't exist
        if not checkpoint_path.is_file():
            print(f"[!] Checkpoint not found at '{checkpoint_path}'. Starting from scratch.")
            return

        print(f"[*] Loading checkpoint: {checkpoint_path}")
        
        # Edge Case 2: Device mapping. Safely loads GPU-saved tensors onto a CPU machine if needed.
        # weights_only=False is required because optimizer/scheduler states contain complex Python objects.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore Core Components
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Advance the epoch by 1 since the saved epoch was already completed
        self.current_epoch = checkpoint.get('epoch', 0) + 1

        # Edge Case 3: Scaler state might be missing if moving from older non-AMP code
        if checkpoint.get('scaler_state') is not None and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state'])
            
        # Edge Case 4: EMA configuration changes
        checkpoint_has_ema = checkpoint.get('ema_state') is not None
        if self.ema_model and checkpoint_has_ema:
            self.ema_model.load_state_dict(checkpoint['ema_state'])
        elif self.ema_model and not checkpoint_has_ema:
            print("[-] Warning: EMA enabled in config, but checkpoint has no EMA state. Starting EMA from current weights.")
            # Initialize EMA with current loaded weights
            self.ema_model.update_parameters(self.model) 
        elif not self.ema_model and checkpoint_has_ema:
            print("[-] Warning: Checkpoint contains EMA state, but EMA is disabled in config. Ignoring EMA weights.")

        print(f"[*] Successfully resumed. Continuing from Epoch {self.current_epoch} (Best Val Loss: {self.best_val_loss:.4f})")

    def visualize(self, inputs: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor, prefix: str):
        """Saves a side-by-side comparison of Input | Target | Prediction."""
        n = min(inputs.size(0), self.config.num_viz_images)
        
        # Concatenate along the height or batch dimension. Here we stack them for a clear grid.
        # Shape becomes (3*n, C, H, W)
        grid_tensors = torch.cat([inputs[:n], targets[:n], predictions[:n]], dim=0)
        
        grid = make_grid(grid_tensors, nrow=n, normalize=True, value_range=(-1, 1))
        save_image(grid, self.config.output_dir / f"{prefix}_ep{self.current_epoch}.png")

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(inputs, labels)
                loss, metrics = self.loss_fn(outputs, inputs.detach())
                loss = loss / self.config.grad_accum_steps
            
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.config.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.ema_model:
                    self.ema_model.update_parameters(self.model)

            total_loss += loss.item() * self.config.grad_accum_steps
            pbar.set_postfix({"loss": f"{loss.item() * self.config.grad_accum_steps:.4f}"})
            
            # Save visualizations on the first batch if hitting the frequency
            if step == 0 and self.config.viz_freq > 0 and self.current_epoch % self.config.viz_freq == 0:
                # We assume output contains the generated image. Adjust indexing if output is a complex dict.
                pred_img = outputs if isinstance(outputs, torch.Tensor) else outputs['pred']
                self.visualize(inputs.detach(), targets.detach(), pred_img.detach(), prefix="train")

        return {"train_loss": total_loss / len(self.train_loader)}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        # Use EMA model for validation if available
        eval_model = self.ema_model.module if self.ema_model else self.model
        eval_model.eval()
        
        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]", leave=False)
        
        for step, batch in enumerate(pbar):
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            with torch.amp.autocast(device_type=self.device.type):
                outputs = eval_model(inputs, labels)
                loss, metrics = self.loss_fn(outputs, labels)
                
            total_loss += loss.item()
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
            
            if step == 0 and self.config.viz_freq > 0 and self.current_epoch % self.config.viz_freq == 0:
                pred_img = outputs if isinstance(outputs, torch.Tensor) else outputs['pred']
                self.visualize(inputs, labels, pred_img, prefix="val")

        return {"val_loss": total_loss / len(self.val_loader)}

    def train(self):
        print(f"Starting training on {self.device} for {self.config.total_epochs} epochs...")
        main_pbar = tqdm(range(self.current_epoch, self.config.total_epochs), desc="Overall Progress")
        
        try:
            for epoch in main_pbar:
                self.current_epoch = epoch
                
                train_metrics = self.train_epoch()
                val_metrics = self.validate()
                self.scheduler.step()
                
                # Combine metrics and log
                epoch_metrics = {
                    "epoch": epoch,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    **train_metrics,
                    **val_metrics
                }
                self.logger.log(epoch_metrics)
                
                # Update main progress bar
                main_pbar.set_postfix({
                    "T_loss": f"{train_metrics['train_loss']:.4f}",
                    "V_loss": f"{val_metrics['val_loss']:.4f}"
                })

                # Checkpointing logic
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint("latest", is_best=True)
                else:
                    self.save_checkpoint("latest", is_best=False)

                if epoch > 0 and epoch % self.config.checkpoint_freq == 0:
                    self.save_checkpoint(f"checkpoint_ep{epoch}")

        except KeyboardInterrupt:
            print("\n[!] Keyboard interrupt received. Saving current state and exiting gracefully...")
            self.save_checkpoint("interrupted_latest")
            print(f"State saved to {self.config.output_dir}/interrupted_latest.pt")