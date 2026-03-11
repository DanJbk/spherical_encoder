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


class Spherefiy(nn.Module):
    def __init__(self, alpha_max=85, alpha_max_ceiling=89, apply_angle_augmentation=True):
        super().__init__()
        self.sigma_max = torch.tan(torch.deg2rad(torch.tensor([alpha_max], device="cpu", dtype=torch.float32))).item()
        self.sigma_max_ceiling = torch.tan(torch.deg2rad(torch.tensor([alpha_max_ceiling], device="cpu", dtype=torch.float32))).item()

        self.apply_angle_augmentation = apply_angle_augmentation

    def forward(self, latents, train=False):
        return self.sphereify(latents, self.sigma_max, train=train)

    def sphereify(self, latents, sigma_max=11, train=False):
        device = latents.device
        latents = F.rms_norm(latents, latents.shape[1:], eps=1e-6)

        if train:
            view_shape = (latents.shape[0],) + (1,) * (len(latents.shape) - 1)
            r = torch.rand(latents.shape[0], device=device).view(view_shape)
            s = torch.rand(latents.shape[0], device=device).view(view_shape) * 0.5
            e = torch.randn_like(latents, device=device)

            sigma = sigma_max * r

            if self.apply_angle_augmentation:
                sigma_augmented = sigma_max + torch.rand(latents.shape[0], device=device).view(view_shape) * (self.sigma_max_ceiling - sigma_max)
                augment_mask = (torch.rand(latents.shape[0], device=device) < 0.1).view(view_shape)
                sigma = torch.where(augment_mask, sigma_augmented, sigma)
        
            sigma_sub = sigma * s
            
            latents_big = F.rms_norm((latents + sigma * e), latents.shape[1:], eps=1e-6)
            latents_small = F.rms_norm((latents + sigma_sub * e), latents.shape[1:], eps=1e-6)

            return latents, latents_big, latents_small
        
        return latents, None, None

    def sample(self, encoder, decoder, latent_shape, class_label=None, cfg_scale=1.0, do_enc_cfg=False, do_dec_cfg=False, T=4, r=1.0, device="cpu"):

        cached_noise = torch.randn(latent_shape, device=device)
        
        if do_dec_cfg or do_enc_cfg:
            assert cfg_scale is not None, "cfg_scale must be provided if using classifier-free guidance"
        
        if do_enc_cfg and do_dec_cfg:
            cfg_scale = cfg_scale**0.5

        v = F.rms_norm(cached_noise, latent_shape[1:], eps=1e-6)
        x = decoder(v, class_label)

        if do_dec_cfg:
            x_uncond = decoder(v, None)
            x = x_uncond + cfg_scale * (x - x_uncond)

        for _ in range(T - 1):
            z = encoder(x, class_label)

            if do_enc_cfg:
                z_uncond = encoder(x, None)
                z = z_uncond + cfg_scale * (z - z_uncond)

            z = F.rms_norm(z, z.shape[1:], eps=1e-6)
            v = F.rms_norm((z + self.sigma_max * r * cached_noise), z.shape[1:], eps=1e-6)

            x = decoder(v, class_label)

            if do_dec_cfg:
                x_uncond = decoder(v, None)
                x = x_uncond + cfg_scale * (x - x_uncond)
                
        return x

# ==========================================
# 3. Core Transformer Blocks
# ==========================================

class SwiGLUFFN(nn.Module):
    """
    Swish-Gated Linear Unit Feed-Forward Network
    """

    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        self.w12 = nn.Linear(dim, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)

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
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
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

        x = F.scaled_dot_product_attention(q, k, v)
        x = self.proj(x.transpose(1, 2).reshape(B, N, C))
        return x

class ViTBlockAdaLNZero(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn = AttentionRoPE(hidden_dim, num_heads=num_heads)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        
        # self.mlp = Mlp(hidden_dim, int(hidden_dim * mlp_ratio))
        self.mlp = SwiGLUFFN(hidden_dim, expansion_factor=2 / 3 * mlp_ratio)
        
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
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.token_mix = Mlp(seq_len, tokens_mlp_dim, seq_len, act_layer=nn.SiLU)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.channel_mix = Mlp(hidden_dim, channels_mlp_dim, hidden_dim, act_layer=nn.SiLU)
        # self.mixer_norm = nn.RMSNorm(hidden_dim, elementwise_affine=True)

        self.alpha1 = nn.Parameter(torch.tensor([0.0]))
        self.alpha2 = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        res = x
        x = self.norm1(x).transpose(1, 2)
        x = self.token_mix(x).transpose(1, 2)
        x = res + self.alpha1 * x
        x = x + self.alpha2 * self.channel_mix(self.norm2(x))
        # x = self.mixer_norm(x)
        return x

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)


class ModulatedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, use_modulation=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_modulation:
            self.norm = nn.RMSNorm(in_features, eps=1e-6)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(in_features, 2 * in_features, bias=bias)
            )
        self.use_modulation = use_modulation

    def forward(self, x, cond=None):
        if self.use_modulation:
            shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
            x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ==========================================
# 4. Sphere Encoder
# ==========================================

class SphereEncoderViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3,
                 hidden_dim=512, depth=8, num_heads=8, num_classes=1000, latent_channels=256,
                 mixer_layers=4, tokens_mlp_ratio=1.0, channels_mlp_ratio=2.0, class_embed_dropout=0.1):
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
        
        self.num_classes = num_classes

        # Class Conditioning (index num_classes is the learned null/unconditional embedding)
        self.class_embed = nn.Embedding(num_classes + 1, hidden_dim)

        # ViT Blocks
        self.vit_blocks = nn.ModuleList([
            ViTBlockAdaLNZero(hidden_dim, num_heads) for _ in range(depth)
        ])

        # MLP-Mixer at the END of the encoder
        self.ffn = ModulatedLinear(hidden_dim, latent_channels, use_modulation=True)

        tokens_mlp_dim = int(self.seq_len * tokens_mlp_ratio)
        channels_mlp_dim = int(latent_channels * channels_mlp_ratio)
        self.mixer_blocks = nn.ModuleList([
            MLPMixerBlock(self.seq_len, latent_channels, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(mixer_layers)
        ])
        self.norm = nn.RMSNorm(latent_channels, eps=1e-6)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        w1 = self.patch_embed[0].weight.data
        nn.init.xavier_uniform_(w1.view(w1.shape[0], -1))
        w2 = self.patch_embed[1].weight.data
        nn.init.xavier_uniform_(w2.view(w2.shape[0], -1))
        nn.init.constant_(self.patch_embed[1].bias, 0)

        nn.init.normal_(self.class_embed.weight, std=0.02)

        for block in self.vit_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.ffn.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.ffn.adaLN_modulation[-1].bias, 0)

    def forward(self, x, class_labels=None):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        c_idx = class_labels if class_labels is not None else torch.full((B,), self.num_classes, device=x.device, dtype=torch.long)
        if class_labels is not None and self.training and self.class_embed_dropout > 0:
            dropout_mask = torch.rand(B, device=x.device) < self.class_embed_dropout
            c_idx = torch.where(dropout_mask, torch.full_like(c_idx, self.num_classes), c_idx)
        c = self.class_embed(c_idx)

        for vit_blk in self.vit_blocks:
            x = vit_blk(x, c, self.freqs_cis)

        x = self.ffn(x, c)
        for mixer_blk in self.mixer_blocks:
            x = mixer_blk(x)
        x = self.norm(x)

        return x

# ==========================================
# 5. Sphere Decoder
# ==========================================

class SphereDecoderViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, out_channels=3,
                 hidden_dim=512, depth=8, num_heads=8, num_classes=1000, latent_channels=256,
                 mixer_layers=4, tokens_mlp_ratio=1.0, channels_mlp_ratio=2.0, class_embed_dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.seq_len = self.grid_size ** 2
        self.class_embed_dropout = class_embed_dropout
        
        # MLP-Mixer at the BEGINNING of the decoder
        tokens_mlp_dim = int(self.seq_len * tokens_mlp_ratio)
        channels_mlp_dim = int(hidden_dim * channels_mlp_ratio)
        self.ffn = nn.Linear(latent_channels, hidden_dim)
        self.norm = nn.RMSNorm(hidden_dim, eps=1e-6)
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
        
        self.num_classes = num_classes

        # Class Conditioning (index num_classes is the learned null/unconditional embedding)
        self.class_embed = nn.Embedding(num_classes + 1, hidden_dim)

        # ViT Blocks
        self.vit_blocks = nn.ModuleList([
            ViTBlockAdaLNZero(hidden_dim, num_heads) for _ in range(depth)
        ])

        # Unpatchify Head
        self.head = ModulatedLinear(hidden_dim, patch_size * patch_size * out_channels, use_modulation=True)
        self.conv_norm_head = nn.Sequential(
            nn.Conv2d(3, 3, stride=1, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.Tanh(),
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # decoder input linear uses normal init (matches reference x_embedder init)
        nn.init.normal_(self.ffn.weight)
        nn.init.constant_(self.ffn.bias, 0)

        nn.init.normal_(self.class_embed.weight, std=0.02)

        for block in self.vit_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)

        # pixel head Conv2d with small gain (matches reference out[-2] init)
        w = self.conv_norm_head[0].weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1), gain=0.01)
        nn.init.constant_(self.conv_norm_head[0].bias, 0)

    def forward(self, x, class_labels=None):
        # Input 'x' expected shape: (B, Seq_Len, Latent_Channels)
        B = x.shape[0]

        x = self.ffn(x)
        for mixer_blk in self.mixer_blocks:
            x = mixer_blk(x)
        x = self.norm(x)

        x = x + self.pos_embed

        c_idx = class_labels if class_labels is not None else torch.full((B,), self.num_classes, device=x.device, dtype=torch.long)
        if class_labels is not None and self.training and self.class_embed_dropout > 0:
            dropout_mask = torch.rand(B, device=x.device) < self.class_embed_dropout
            c_idx = torch.where(dropout_mask, torch.full_like(c_idx, self.num_classes), c_idx)
        c = self.class_embed(c_idx)

        for vit_blk in self.vit_blocks:
            x = vit_blk(x, c, self.freqs_cis)

        x = self.head(x, c)

        # Fold back into Image Format (channel-first within patch, matching reference ordering)
        x = x.reshape(B, self.grid_size, self.grid_size, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, -1, self.img_size, self.img_size)

        x = self.conv_norm_head(x)

        return x


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.args = kwargs.copy()

        decoder_kwargs = kwargs.copy()
        decoder_kwargs["out_channels"] = decoder_kwargs.pop("in_channels")
        
        self.encoder = SphereEncoderViT(**kwargs)
        self.decoder = SphereDecoderViT(**decoder_kwargs)

        self.spherefiy = Spherefiy()

    def forward(self, x, labels, cfg_scale=2.0, do_enc_cfg=True, do_dec_cfg=True, T=4, r=1.0):
        if True:
            latents_cond = self.encoder(x, class_labels=labels)
            spherified_latents_cond, noisy, less_noisy = self.spherefiy(latents_cond, train=True) # (Placeholder)

            x_NOISY = self.decoder(noisy, class_labels=labels)
            x_noisy = self.decoder(less_noisy, class_labels=labels)

            x_noisy_sg = x_noisy.detach().clone()
            v_one_step =  self.encoder(x_NOISY, class_labels=labels)

            return x, spherified_latents_cond, x_NOISY, x_noisy, x_noisy_sg, v_one_step
        
        else:
            latent_channel_size = self.args["latent_channels"]
            return self.spherefiy.sample(
                self.encoder, 
                self.decoder, 
                latent_shape=(1, latent_channel_size, latent_channel_size), 
                class_label=labels, cfg_scale=cfg_scale, do_enc_cfg=do_enc_cfg, 
                do_dec_cfg=do_dec_cfg, 
                T=T, 
                r=r, 
                device=x.device
            )

if __name__ == "__main__":
    # --- Configuration ---
    # batch_size = 2
    # # img_size = 256
    # img_size = 32
    # patch_size = 16
    # in_channels = 3
    # hidden_dim = 512
    # latent_channels = 256
    # num_classes = 1000
    # num_heads = 8
    # depth = 8

    batch_size = 2
    img_size = 32
    patch_size = 2
    in_channels = 3
    hidden_dim = 384
    latent_channels = 8
    num_classes = 10
    num_heads = 6
    depth = 12


    print("Instantiating Sphere Encoder and Decoder...")
    encoder = SphereEncoderViT(
        img_size=img_size, 
        patch_size=patch_size, 
        in_channels=in_channels, 
        hidden_dim=hidden_dim, 
        latent_channels=latent_channels,
        num_classes=num_classes,
        num_heads=num_heads,
        depth=depth,
    )
    
    decoder = SphereDecoderViT(
        img_size=img_size, 
        patch_size=patch_size, 
        out_channels=in_channels, 
        hidden_dim=hidden_dim, 
        latent_channels=latent_channels,
        num_classes=num_classes,
        num_heads=num_heads,
        depth=depth,
    )

    spherefiy = Spherefiy()

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
    spherified_latents_cond, noisy, less_noisy = spherefiy(latents_cond, train=True) # (Placeholder)
    print(f"{noisy.shape=}")
    print(f"{less_noisy.shape=}")

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
    spherified_latents_uncond, _, _ = spherefiy(latents_cond) # (Placeholder)
    
    recon_uncond = decoder(spherified_latents_uncond, class_labels=None)
    print(f"Unconditional Decoder Output Shape: {recon_uncond.shape}")

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")

    seq_len = (img_size // patch_size) ** 2
    print(f"latent shape: {(2, seq_len, latent_channels)}")
    samples = spherefiy.sample(
        encoder, 
        decoder, 
        latent_shape=(2, seq_len, latent_channels), 
        class_label=torch.tensor([42, 11]), 
        cfg_scale=2.0, do_enc_cfg=True, 
        do_dec_cfg=True, 
        T=4, 
        r=1.0, 
        device="cpu"
    )
    print(samples.shape)
