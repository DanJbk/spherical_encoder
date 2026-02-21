import torch
import torch.nn as nn
import math

def modulate(x, shift, scale):
    """Applies AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Mlp(nn.Module):
    """Standard MLP for ViT."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Attention(nn.Module):
    """Standard Multi-Head Self Attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ViTBlockAdaLNZero(nn.Module):
    """ViT Block with AdaLN-Zero conditioning (DiT style)."""
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_dim, int(hidden_dim * mlp_ratio))
        
        # AdaLN-Zero modulation: outputs shift, scale, and gate for both MSA and MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        
        # Zero-initialize the last linear layer
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention branch
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # MLP branch
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class MLPMixerBlock(nn.Module):
    """Standard MLP-Mixer block."""
    def __init__(self, seq_len, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mix = Mlp(seq_len, tokens_mlp_dim, seq_len)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mix = Mlp(hidden_dim, channels_mlp_dim, hidden_dim)

    def forward(self, x):
        # Token mixing (acts across spatial sequence)
        res = x
        x = self.norm1(x).transpose(1, 2)
        x = self.token_mix(x).transpose(1, 2)
        x = x + res
        
        # Channel mixing (acts across hidden dimension)
        x = x + self.channel_mix(self.norm2(x))
        return x

class ConditionalViTEncoder(nn.Module):
    """
    Standard ViT with optional MLP-Mixer and AdaLN-Zero class conditioning for generation.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        hidden_dim=768,
        depth=12,
        num_heads=12,
        num_classes=1000,
        use_mixer=True,
        mixer_position='end', # 'start' or 'end'
        mixer_layers=4,
        tokens_mlp_ratio=0.5,
        channels_mlp_ratio=4.0
    ):
        super().__init__()
        assert mixer_position in ['start', 'end'], "mixer_position must be 'start' or 'end'"
        
        self.hidden_dim = hidden_dim
        self.use_mixer = use_mixer
        self.mixer_position = mixer_position
        
        # 1. Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        seq_len = (img_size // patch_size) ** 2
        
        # 2. Positional Embedding (standard learnable, omitting CLS token for generation tasks)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        
        # 3. Class Conditioning (Separate embeddings + Null for CFG)
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.null_class_embed = nn.Parameter(torch.zeros(1, hidden_dim)) # For CFG dropouts
        
        # 4. Optional MLP-Mixer
        if use_mixer:
            tokens_mlp_dim = int(seq_len * tokens_mlp_ratio)
            channels_mlp_dim = int(hidden_dim * channels_mlp_ratio)
            self.mixer_blocks = nn.ModuleList([
                MLPMixerBlock(seq_len, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
                for _ in range(mixer_layers)
            ])
            
        # 5. ViT Blocks with AdaLN-Zero
        self.vit_blocks = nn.ModuleList([
            ViTBlockAdaLNZero(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d default)
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.null_class_embed, std=0.02)

    def forward(self, x, class_labels=None):
        """
        x: (B, C, H, W)
        class_labels: (B,) class indices. If None, forces unconditional generation (CFG).
                      Alternatively, you can pass a batched vector of indices with specific
                      indices dropped via training logic before passing to the model.
        """
        B = x.shape[0]
        
        # Patching
        x = self.patch_embed(x)      # (B, hidden_dim, H_p, W_p)
        x = x.flatten(2).transpose(1, 2) # (B, seq_len, hidden_dim)
        
        # Add Positional Embedding
        x = x + self.pos_embed
        
        # Class Conditioning Setup
        if class_labels is not None:
            c = self.class_embed(class_labels) # (B, hidden_dim)
        else:
            # Broadcast learned null embedding for CFG unconditionally
            c = self.null_class_embed.expand(B, -1)
            
        # Optional MLP-Mixer at the START
        if self.use_mixer and self.mixer_position == 'start':
            for mixer_blk in self.mixer_blocks:
                x = mixer_blk(x)
                
        # ViT Blocks
        for vit_blk in self.vit_blocks:
            x = vit_blk(x, c)
            
        # Optional MLP-Mixer at the END
        if self.use_mixer and self.mixer_position == 'end':
            for mixer_blk in self.mixer_blocks:
                x = mixer_blk(x)
                
        # To match DiT/AdaLN-Zero setups, final norm is often unmodulated or conditionally modulated.
        # Here we apply standard LayerNorm.
        x = self.final_norm(x)
        
        return x

# Example Usage:
if __name__ == "__main__":
    batch_size = 2
    img_size = 256
    model = ConditionalViTEncoder(img_size=img_size, mixer_position='end')
    
    # Dummy Image
    dummy_x = torch.randn(batch_size, 3, img_size, img_size)
    
    # Conditional forward (with labels)
    labels = torch.randint(0, 1000, (batch_size,))
    out_cond = model(dummy_x, class_labels=labels)
    
    # Unconditional forward (CFG null embedding triggered)
    out_uncond = model(dummy_x, class_labels=None)
    
    print(f"Output shape: {out_cond.shape}") # Expected: (2, 256, 768)