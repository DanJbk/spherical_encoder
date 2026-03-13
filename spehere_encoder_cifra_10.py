import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from trainer.trainer import ImageToImageTrainer
from trainer.config import TrainerConfig, CSVLogger
from models.vit import Model
from losses.loss import CombinedLoss

# Dataset Wrapper (Image-to-Image for CIFAR-10)

class CIFAR10AutoencoderDataset(Dataset):
    def __init__(self, root: str = "./data", train: bool = True):
        # We normalize to [-1, 1] because our trainer's visualization 
        # uses value_range=(-1, 1) in make_grid.
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        
        
        self.cifar = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, label = self.cifar[idx] # We discard the label
        # Return (input, target). For a basic autoencoder, they are identical.
        return img, label


# Execution setup

if __name__ == "__main__":
    # Setup data
    train_dataset = CIFAR10AutoencoderDataset(train=True)
    val_dataset = CIFAR10AutoencoderDataset(train=False)
    
    
    sphere_loss = CombinedLoss()

    # Setup model and config
    batch_size = 64
    img_size = 32
    patch_size = 2
    in_channels = 3
    hidden_dim = 384
    latent_channels = 8
    num_classes = 10
    num_heads = 6
    depth = 12

    resume_checkpoint = False
    output_dir = Path("runs11/cifar10_test")

    model = Model(
        img_size=img_size, 
        patch_size=patch_size, 
        in_channels=in_channels, 
        hidden_dim=hidden_dim, 
        latent_channels=latent_channels,
        num_classes=num_classes,
        num_heads=num_heads,
        depth=depth,
    )

    config = TrainerConfig(
        batch_size=batch_size,
        grad_accum_steps=1,
        ema_decay=None,
        total_epochs=100,     # Just 5 epochs for a quick test
        warmup_epochs=10,
        viz_freq=0,           # Save images every epoch so you can see it work instantly
        checkpoint_freq=2,
        output_dir=output_dir

    )
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=5, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=5, persistent_workers=True)

    if output_dir.exists() and not resume_checkpoint:
        print(f"Output directory {output_dir} already exists. Please remove it or choose a different path to avoid overwriting.")
        exit(1)
    
    # Initialize and run
    trainer = ImageToImageTrainer(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        loss_fn=sphere_loss,
        config=config
    )
    
    if resume_checkpoint:
        latest_checkpoint = config.output_dir / "checkpoint_ep34.pt"
        trainer.resume_from_checkpoint(latest_checkpoint)
    
    trainer.train()