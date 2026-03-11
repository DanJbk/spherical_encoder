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

@dataclass
class TrainerConfig:
    # Training defaults
    image_size: int = 32
    batch_size: int = 64
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    warmup_epochs: int = 10
    total_epochs: int = 5000
    
    # Newly added features
    grad_accum_steps: int = 1
    ema_decay: float | None = None  # e.g., 0.999. None means disabled.
    
    # I/O and Checkpointing
    checkpoint_freq: int = 100
    viz_freq: int = 50          # How often to save image grids
    num_viz_images: int = 8     # Number of images to include in the grid
    output_dir: Path = field(default_factory=lambda: Path("runs/experiment"))
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

class CSVLogger:
    """Simple logger that appends metrics to a CSV file."""
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.headers_written = self.filepath.exists()

    def log(self, metrics: dict[str, float | int]):
        mode = 'a' if self.filepath.exists() else 'w'
        with open(self.filepath, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not self.headers_written and mode == 'w':
                writer.writeheader()
                self.headers_written = True
            writer.writerow(metrics)