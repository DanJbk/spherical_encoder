import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Utility & Normalization Modules
# ==========================================

def modulate(x, shift, scale):
    """Applies AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ==========================================
# 2. Positional Embeddings (Absolute & RoPE)
# ==========================================

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Generates standard 2D sinusoidal absolute positional encoding."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= (embed_dim / 2.)
        omega = 1. / 10000**omega
        pos = pos.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=1)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)

def precompute_freqs_cis_2d(grid_size, head_dim):
    """Precomputes 2D RoPE frequencies for the Attention mechanism."""
    half_dim = head_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
    
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    
    freqs_h = torch.outer(grid_h, omega)
    freqs_w = torch.outer(grid_w, omega)
    
    freqs_h = torch.polar(torch.ones_like(freqs_h), freqs_h)
    freqs_w = torch.polar(torch.ones_like(freqs_w), freqs_w)
    
    freqs_h = freqs_h.view(grid_size, 1, -1).expand(grid_size, grid_size, -1)
    freqs_w = freqs_w.view(1, grid_size, -1).expand(grid_size, grid_size, -1)
    
    freqs_cis = torch.cat([freqs_h, freqs_w], dim=-1).reshape(grid_size * grid_size, -1)
    return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    """Applies Rotary Position Embedding to Queries and Keys."""
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (1, SeqLen, 1, head_dim//2)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


def sphereify(latents, sigma=1, noise=None):
    latents = F.rms_norm(latents, latents.shape[1:], eps=1e-6)

    if noise:
        view_shape = [latents.shape[0]] + [1] * (latents.ndim - 1)
        
        sigma = (sigma * torch.rand(latents.shape[0])).view(*view_shape)
        noise_latent = torch.randn_like(latents)
        latents = F.rms_norm((latents + sigma * noise_latent), latents.shape[1:], eps=1e-6)
    
    return latents

# ==========================================
# 3. Core Transformer Blocks
# ==========================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class AttentionRoPE(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ViTBlockAdaLNZero(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionRoPE(hidden_dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_dim, int(hidden_dim * mlp_ratio))
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x, c, freqs_cis):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class MLPMixerBlock(nn.Module):
    def __init__(self, seq_len, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mix = Mlp(seq_len, tokens_mlp_dim, seq_len)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mix = Mlp(hidden_dim, channels_mlp_dim, hidden_dim)
        self.mixer_norm = nn.RMSNorm(hidden_dim, elementwise_affine=True)

    def forward(self, x):
        res = x
        x = self.norm1(x).transpose(1, 2)
        x = self.token_mix(x).transpose(1, 2)
        x = x + res
        x = x + self.channel_mix(self.norm2(x))
        x = self.mixer_norm(x)
        return x

# ==========================================
# 4. Sphere Encoder
# ==========================================

class SphereEncoderViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3,
                 hidden_dim=1024, depth=24, num_heads=16, num_classes=1000, latent_channels=16,
                 mixer_layers=4, tokens_mlp_ratio=0.5, channels_mlp_ratio=4.0, class_embed_dropout=0.1):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.seq_len = self.grid_size ** 2
        self.class_embed_dropout = class_embed_dropout
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            )
        
        # Absolute Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, hidden_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(hidden_dim, self.grid_size)
        self.pos_embed.data.copy_((pos_embed).float().unsqueeze(0))
        
        # RoPE Frequencies
        self.register_buffer("freqs_cis", precompute_freqs_cis_2d(self.grid_size, hidden_dim // num_heads))
        
        # Class Conditioning
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.null_class_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        
        # ViT Blocks
        self.vit_blocks = nn.ModuleList([
            ViTBlockAdaLNZero(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        # MLP-Mixer at the END of the encoder
        self.ffn = nn.Linear(hidden_dim, latent_channels) # todo replace with modulated linear layer

        tokens_mlp_dim = int(self.seq_len * tokens_mlp_ratio)
        channels_mlp_dim = int(latent_channels * channels_mlp_ratio)
        self.mixer_blocks = nn.ModuleList([
            MLPMixerBlock(self.seq_len, latent_channels, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(mixer_layers)
        ])

        # self.ffn = ModulatedLinear(
        #         self.hidden_size, intermediate_chns, use_modulation=self.use_modulation
        #     )
        
        self.initialize_weights()

    def initialize_weights(self):
        w1 = self.patch_embed[0].weight.data
        nn.init.xavier_uniform_(w1.view(w1.shape[0], -1))
        w2 = self.patch_embed[1].weight.data
        nn.init.xavier_uniform_(w2.view(w2.shape[0], -1))
        nn.init.constant_(self.patch_embed[1].bias, 0)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.null_class_embed, std=0.02)

    def forward(self, x, class_labels=None):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        c = self.class_embed(class_labels) if class_labels is not None else self.null_class_embed.expand(B, -1)
        if class_labels is not None and self.training and self.class_embed_dropout > 0:
            dropout_mask = torch.rand(B, device=x.device) < self.class_embed_dropout
            c = torch.where(dropout_mask.unsqueeze(1), self.null_class_embed.expand(B, -1), c)
        
        for vit_blk in self.vit_blocks:
            x = vit_blk(x, c, self.freqs_cis)
            
        x = self.ffn(x)
        for mixer_blk in self.mixer_blocks:
            x = mixer_blk(x)

            
        return x

# ==========================================
# 5. Sphere Decoder
# ==========================================

class SphereDecoderViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, out_channels=3,
                 hidden_dim=1024, depth=24, num_heads=16, num_classes=1000, latent_channels=16,
                 mixer_layers=4, tokens_mlp_ratio=0.5, channels_mlp_ratio=4.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.seq_len = self.grid_size ** 2
        
        # MLP-Mixer at the BEGINNING of the decoder
        tokens_mlp_dim = int(self.seq_len * tokens_mlp_ratio)
        channels_mlp_dim = int(hidden_dim * channels_mlp_ratio)
        self.ffn = nn.Linear(latent_channels, hidden_dim)
        self.mixer_blocks = nn.ModuleList([
            MLPMixerBlock(self.seq_len, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(mixer_layers)
        ])
        
        # Absolute Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, hidden_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(hidden_dim, self.grid_size)
        self.pos_embed.data.copy_((pos_embed).float().unsqueeze(0))
        
        # RoPE Frequencies
        self.register_buffer("freqs_cis", precompute_freqs_cis_2d(self.grid_size, hidden_dim // num_heads))
        
        # Class Conditioning
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.null_class_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        
        # ViT Blocks
        self.vit_blocks = nn.ModuleList([
            ViTBlockAdaLNZero(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # Unpatchify Head
        self.head = nn.Linear(hidden_dim, patch_size * patch_size * out_channels) # todo replace with conv2d head
        
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.null_class_embed, std=0.02)

    def forward(self, x, class_labels=None):
        # Input 'x' expected shape: (B, Seq_Len, Latent_Channels)
        B = x.shape[0]
        
        x = self.ffn(x)
        for mixer_blk in self.mixer_blocks:
            x = mixer_blk(x)
            
        x = x + self.pos_embed
        
        c = self.class_embed(class_labels) if class_labels is not None else self.null_class_embed.expand(B, -1)
        
        for vit_blk in self.vit_blocks:
            x = vit_blk(x, c, self.freqs_cis)
            
        x = self.final_norm(x)
        x = self.head(x)
        
        # Fold back into Image Format
        x = x.reshape(B, self.grid_size, self.grid_size, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, self.img_size, self.img_size)

        return x
    

if __name__ == "__main__":
    # --- Configuration ---
    batch_size = 2
    img_size = 256
    patch_size = 16
    in_channels = 3
    hidden_dim = 512
    num_classes = 1000
    num_heads = 8
    depth = 8

    print("Instantiating Sphere Encoder and Decoder...")
    encoder = SphereEncoderViT(
        img_size=img_size, 
        patch_size=patch_size, 
        in_channels=in_channels, 
        hidden_dim=hidden_dim, 
        num_classes=num_classes,
        num_heads=num_heads,
        depth=depth,
    )
    
    decoder = SphereDecoderViT(
        img_size=img_size, 
        patch_size=patch_size, 
        out_channels=in_channels, 
        hidden_dim=hidden_dim, 
        num_classes=num_classes,
        num_heads=num_heads,
        depth=depth,
    )

    # --- Dummy Inputs ---
    dummy_img = torch.randn(batch_size, in_channels, img_size, img_size)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))

    print(f"Dummy Labels: {dummy_labels.shape}")

    print(f"Input Image Shape: {dummy_img.shape}")

    # ==========================================
    # Conditional Forward Pass
    # ==========================================
    print("\n--- Conditional Pass ---")
    # 1. Encode
    latents_cond = encoder(dummy_img, class_labels=dummy_labels)
    print(f"Encoder Output (Latents) Shape: {latents_cond.shape}") 
    # Expected: (2, 256, 1024) where 256 is the seq_len (256//16)^2

    # ---> YOUR SPHERIFICATION LOGIC GOES HERE <---
    # e.g., v = f(latents_cond + sigma * e)
    spherified_latents_cond = latents_cond # (Placeholder)

    # 2. Decode
    recon_cond = decoder(spherified_latents_cond, class_labels=dummy_labels)
    print(f"Decoder Output (Reconstruction) Shape: {recon_cond.shape}") 
    # Expected: (2, 3, 256, 256)

    # ==========================================
    # Unconditional Forward Pass (CFG)
    # ==========================================
    print("\n--- Unconditional Pass (CFG) ---")
    # Setting class_labels=None triggers the learned null embedding
    latents_uncond = encoder(dummy_img, class_labels=None)
    
    # ---> YOUR SPHERIFICATION LOGIC GOES HERE <---
    spherified_latents_uncond = latents_uncond # (Placeholder)
    
    recon_uncond = decoder(spherified_latents_uncond, class_labels=None)
    print(f"Unconditional Decoder Output Shape: {recon_uncond.shape}")

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")

