import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        return x


class MlpMixer(nn.Module):
    def __init__(self, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, image_height=360, image_width=640):
        super(MlpMixer, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_tokens = (image_height // patch_size) * (image_width // patch_size)

        self.patch_emb = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.mlp = nn.Sequential(*[MixerBlock(self.num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_emb(x)  # Shape: [B, hidden_dim, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, num_tokens, hidden_dim]

        # MLP-Mixer Blocks
        x = self.mlp(x)

        # LayerNorm
        x = self.ln(x)  # Shape: [B, num_tokens, hidden_dim]


            # Reshape back to B, C, H, W
        x = x.transpose(1, 2)  # Shape: [B, hidden_dim, num_tokens]
        x = x.view(B, self.hidden_dim, H, W)  # Shape: [B, C, H, W]
        return x
