import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from trainer.trainer import ImageToImageTrainer
from trainer.config import TrainerConfig, CSVLogger

# ---------------------------------------------------------
# 1. Dataset Wrapper (Image-to-Image for CIFAR-10)
# ---------------------------------------------------------
class CIFAR10AutoencoderDataset(Dataset):
    def __init__(self, root: str = "./data", train: bool = True):
        # We normalize to [-1, 1] because our trainer's visualization 
        # uses value_range=(-1, 1) in make_grid.
        transform = transforms.Compose([
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

# ---------------------------------------------------------
# 2. Dummy Model (Matching our custom output requirement)
# ---------------------------------------------------------
class DummyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x, y):
        latent = self.encoder(x)
        out = self.decoder(latent)
        
        # Return a dictionary instead of a tuple!
        # The trainer specifically looks for the "pred" key to save the image.
        return {
            "pred": out, 
            "targets": x,
            "latent": latent.view(latent.size(0), -1)
        }

# ---------------------------------------------------------
# 3. Custom Loss Function
# ---------------------------------------------------------
def custom_loss_fn(outputs: dict[str, torch.Tensor], labels: torch.Tensor):
    # Unpack from the dictionary
    pred_images = outputs["pred"]
    latent_vectors = outputs["latent"]
    targets = outputs["targets"]
    
    # Primary loss
    mse_loss = nn.functional.mse_loss(pred_images, targets)
    
    # Auxiliary loss
    aux_loss = torch.mean(latent_vectors ** 2)
    
    # Combine them
    total_loss = mse_loss + (0.01 * aux_loss)
    metrics_dict = {"mse": mse_loss.detach(), "aux": aux_loss.detach()}
    
    return total_loss, metrics_dict

# ---------------------------------------------------------
# 4. Execution setup
# ---------------------------------------------------------
if __name__ == "__main__":
    # Setup data
    train_dataset = CIFAR10AutoencoderDataset(train=True)
    val_dataset = CIFAR10AutoencoderDataset(train=False)
    
    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # Setup model and config
    model = DummyAutoencoder()
    config = TrainerConfig(
        batch_size=64,
        grad_accum_steps=1,
        ema_decay=0.999,
        total_epochs=5,       # Just 5 epochs for a quick test
        warmup_epochs=1,
        viz_freq=0,           # Save images every epoch so you can see it work instantly
        checkpoint_freq=2,
        output_dir=Path("runs/cifar10_test")
    )
    
    # Initialize and run!
    trainer = ImageToImageTrainer(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        loss_fn=custom_loss_fn,
        config=config
    )
    
    trainer.train()