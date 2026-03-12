
import torch
from torch import nn


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
