import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math

# https://gitlab.com/lucidrains/vit-pytorch/-/blob/main/vit_pytorch/vit.py?ref_type=heads
# helpers

# pos encoddings

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input.
    This implements the transformation: [-x2, x1]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RoPE2D(nn.Module):
    def __init__(self, head_dim: int, max_h: int = 100, max_w: int = 100, base: float = 10000.0):
        """
        Args:
            head_dim (int): The dimension of each attention head. Must be divisible by 4.
            max_h (int): Maximum expected height of the feature grid.
            max_w (int): Maximum expected width of the feature grid.
            base (float): The base for the frequency scaling (default: 10000.0).
        """
        super().__init__()
        # We split the head dimension in two (Y and X axes), 
        # and each axis needs an even number of features for complex rotation.
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4. Got {head_dim}")
            
        self.head_dim = head_dim
        self.half_dim = head_dim // 2 
        
        # 1. Compute frequencies for a single axis
        # Shape: (half_dim // 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        
        # 2. Create the grid coordinates
        grid_h = torch.arange(max_h, dtype=torch.float32)
        grid_w = torch.arange(max_w, dtype=torch.float32)
        
        # 3. Compute the angles (outer product of grid and frequencies)
        # Shape: (max_h, half_dim // 2) and (max_w, half_dim // 2)
        freqs_h = torch.outer(grid_h, inv_freq)
        freqs_w = torch.outer(grid_w, inv_freq)
        
        # 4. Duplicate frequencies to match the rotate_half concatenation logic
        # Shape: (max_h, half_dim) and (max_w, half_dim)
        freqs_h = torch.cat((freqs_h, freqs_h), dim=-1)
        freqs_w = torch.cat((freqs_w, freqs_w), dim=-1)
        
        # 5. Broadcast across the 2D grid
        # freqs_h shape: (max_h, max_w, half_dim)
        # freqs_w shape: (max_h, max_w, half_dim)
        freqs_h = freqs_h.unsqueeze(1).expand(-1, max_w, -1)
        freqs_w = freqs_w.unsqueeze(0).expand(max_h, -1, -1)
        
        # 6. Precompute sine and cosine
        self.register_buffer('cos_h', freqs_h.cos())
        self.register_buffer('sin_h', freqs_h.sin())
        self.register_buffer('cos_w', freqs_w.cos())
        self.register_buffer('sin_w', freqs_w.sin())

    def forward(self, q: torch.Tensor, k: torch.Tensor, h: int, w: int):
        """
        Applies 2D RoPE to the query and key tensors.
        
        Args:
            q (torch.Tensor): Query tensor of shape (Batch, Num_Heads, Seq_Len, Head_Dim)
            k (torch.Tensor): Key tensor of shape (Batch, Num_Heads, Seq_Len, Head_Dim)
            h (int): Height of the current image/feature map
            w (int): Width of the current image/feature map
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors.
        """
        seq_len = h * w
        if seq_len != q.shape[-2]:
            raise ValueError(f"Feature map dimensions h={h}, w={w} do not match sequence length {q.shape[-2]}")
            
        # 1. Slice the precomputed buffers to the current grid size and flatten
        # Shape after reshape: (seq_len, half_dim)
        cos_h = self.cos_h[:h, :w, :].reshape(seq_len, -1)
        sin_h = self.sin_h[:h, :w, :].reshape(seq_len, -1)
        cos_w = self.cos_w[:h, :w, :].reshape(seq_len, -1)
        sin_w = self.sin_w[:h, :w, :].reshape(seq_len, -1)
        
        # 2. Add dimensions to broadcast against (Batch, Num_Heads, Seq_Len, Half_Dim)
        cos_h = cos_h.unsqueeze(0).unsqueeze(0)
        sin_h = sin_h.unsqueeze(0).unsqueeze(0)
        cos_w = cos_w.unsqueeze(0).unsqueeze(0)
        sin_w = sin_w.unsqueeze(0).unsqueeze(0)
        
        # 3. Split queries and keys into Y (height) and X (width) components
        q_h, q_w = q.chunk(2, dim=-1)
        k_h, k_w = k.chunk(2, dim=-1)
        
        # 4. Apply the rotary transformation
        q_h_rotated = (q_h * cos_h) + (rotate_half(q_h) * sin_h)
        q_w_rotated = (q_w * cos_w) + (rotate_half(q_w) * sin_w)
        
        k_h_rotated = (k_h * cos_h) + (rotate_half(k_h) * sin_h)
        k_w_rotated = (k_w * cos_w) + (rotate_half(k_w) * sin_w)
        
        # 5. Concatenate the rotated components back together
        q_rotated = torch.cat([q_h_rotated, q_w_rotated], dim=-1)
        k_rotated = torch.cat([k_h_rotated, k_w_rotated], dim=-1)
        
        return q_rotated, k_rotated


class Absolute2DPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_h: int = 100, max_w: int = 100):
        """
        Args:
            d_model (int): The total embedding dimension. Must be a multiple of 4.
            max_h (int): Maximum expected height of the grid/image.
            max_w (int): Maximum expected width of the grid/image.
        """
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be a multiple of 4. Got {d_model}")
        
        # Split dimension in half for x and y
        d_pos = d_model // 2
        
        # 1. Create y and x position coordinates
        y_pos = torch.arange(max_h, dtype=torch.float32).unsqueeze(1) # Shape: (max_h, 1)
        x_pos = torch.arange(max_w, dtype=torch.float32).unsqueeze(1) # Shape: (max_w, 1)
        
        # 2. Create the division term based on the formula: 1 / (10000 ** (2i / d_pos))
        div_term = torch.exp(torch.arange(0, d_pos, 2, dtype=torch.float32) * (-math.log(10000.0) / d_pos))
        
        # 3. Calculate 1D positional encodings for y and x separately
        pe_y = torch.zeros(max_h, d_pos)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)
        
        pe_x = torch.zeros(max_w, d_pos)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)
        
        # 4. Broadcast to the 2D grid shape: (max_h, max_w, d_pos)
        pe_y = pe_y.unsqueeze(1).expand(-1, max_w, -1)
        pe_x = pe_x.unsqueeze(0).expand(max_h, -1, -1)
        
        # 5. Concatenate along the embedding dimension
        pe = torch.cat([pe_y, pe_x], dim=-1) # Shape: (max_h, max_w, d_model)
        
        # Register as a buffer so it saves with the model state_dict but isn't a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, h: int, w: int) -> torch.Tensor:
        """
        Args:
            h (int): Current height of the input feature map.
            w (int): Current width of the input feature map.
            
        Returns:
            torch.Tensor: Positional encoding of shape (1, h * w, d_model)
        """
        if h > self.pe.shape[0] or w > self.pe.shape[1]:
            raise ValueError(f"Requested size ({h}, {w}) exceeds initialized max size ({self.pe.shape[0]}, {self.pe.shape[1]}).")
        
        # Slice the pre-computed grid to match the current input size
        pe_slice = self.pe[:h, :w, :]
        
        # Flatten the spatial dimensions to match transformer sequence input: (batch, seq_len, d_model)
        # We add a batch dimension of 1 which can be broadcasted across your actual batch size.
        return pe_slice.reshape(1, h * w, -1)


# classes
class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., pos_embeddings = None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img):
        batch = img.shape[0]
        # x = self.to_patch_embedding(img)
        x = apply_rotary_emb(x, freqs_cis)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.mlp_head is None:
            return x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
