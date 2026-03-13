import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class L1PerceptualLoss(nn.Module):
    def __init__(self, wl1, wperc, perceptual):
        super().__init__()

        self.perceptual = perceptual
        
        
        self.wl1 = wl1
        self.wperc = wperc

    def forward(self, predicted: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l1 = F.smooth_l1_loss(predicted, targets)
        
        # The raw net returns a batch vector, so we explicitly .mean() it
        perc = self.perceptual(predicted, targets).mean()
        
        return (self.wl1 * l1) + (self.wperc * perc)

def latent_consistency_loss(z1: torch.Tensor, z2: torch.Tensor, weight=0.1) -> torch.Tensor:

    z1 = z1.reshape(z1.shape[0], -1)
    z2 = z2.reshape(z2.shape[0], -1)
    
    sim = F.cosine_similarity(z1, z2, dim=1)
    return weight * (1.0 - sim.mean())


class CombinedLoss(nn.Module):

    def __init__(self, wl1_recon=1.0, wperc_recon=1.0, wl1_con=0.5, wperc_con=0.5, latent_consistency_weight=0.1, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.perceptual = LearnedPerceptualImagePatchSimilarity(net_type='vgg').net
        self.perceptual.requires_grad_(False)
        self.perceptual.eval()

        self.preceptual_loss_pix_recon = L1PerceptualLoss(wl1=1.0, wperc=1.0, perceptual=self.perceptual).to(device)
        self.preceptual_loss_pix_con = L1PerceptualLoss(wl1=0.5, wperc=0.5, perceptual=self.perceptual).to(device)


    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor):
        l_pix_recon = self.preceptual_loss_pix_recon(outputs["x_noisy"], outputs["x"])
        l_pix_con = self.preceptual_loss_pix_con(outputs["x_NOISY"], outputs["x_noisy_sg"])
        cosine_similarity_loss_value = latent_consistency_loss(outputs["spherified_latents_cond"], outputs["v_one_step"])

        total_loss = l_pix_recon + l_pix_con + cosine_similarity_loss_value * 0.1
        
        return total_loss, {"l_pix_recon": l_pix_recon.item(), "l_pix_con": l_pix_con.item(), "cosine_similarity_loss": cosine_similarity_loss_value.item()}