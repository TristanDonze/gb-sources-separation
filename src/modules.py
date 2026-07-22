import torch
import math
from collections import OrderedDict
import torch.nn as nn

class PermutedLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        # x: (B, C, F)
        x = x.permute(0, 2, 1)  # (B, F, C)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)  # (B, C, F)
        return x

class ConvEncoder(nn.Module):
    def __init__(self, input_channels: int = 4, dim_model: int = 256, dropout: float = 0.1):
        super().__init__()

        self.dim_model = dim_model

        channels = [input_channels, 32] 
        while True:
            next_channel = channels[-1] * 2
            if next_channel >= dim_model:
                break
            channels.append(next_channel)
        channels.append(dim_model)

        self.conv_encoder = nn.Sequential(OrderedDict([]))

        for i in range(len(channels) - 1):
            self.conv_encoder.add_module(f'conv_{i+1}', nn.Conv1d(channels[i], channels[i+1], kernel_size = 7 if i == 0 else 5, padding="same"))
            self.conv_encoder.add_module(f'layernorm_{i+1}', PermutedLayerNorm(channels[i+1]))
            self.conv_encoder.add_module(f'gelu_{i+1}', nn.GELU())
            self.conv_encoder.add_module(f'dropout_{i+1}', nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be (B, L, C, F) or (B, C, F)
        if x.dim() == 4:
            B, L, C, F = x.shape
            x = x.reshape(B * L, C, F)  # Reshape to (B*L, C, F)
            x = self.conv_encoder(x)
            x = x.reshape(B, L, self.dim_model, F)  # Reshape back to (B, L, dim_model, F)
        elif x.dim() == 3:
            x = self.conv_encoder(x)
        else:
            raise ValueError("Input tensor must be 3D or 4D.")

        return x

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int = 256,
        max_period: int = 10_000,
        time_scale: float = 1_000.0,
    ):
        super().__init__()

        half_dim = dim_model // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )

        self.register_buffer("frequencies", frequencies)
        self.dim_model = dim_model
        self.time_scale = time_scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) ou (B, 1)
        t = t.reshape(-1) * self.time_scale

        angles = t[:, None] * self.frequencies[None, :]

        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=-1,
        )

        return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, dim_model: int = 256):
        super().__init__()

        self.sinusoidal_embedding = SinusoidalTimeEmbedding(dim_model)

        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.GELU(),
            nn.Linear(dim_model, dim_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal_embedding(t))

class FiLM(nn.Module):
    def __init__(self, dim_model: int = 256):
        super().__init__()

        self.modulation = nn.Linear(dim_model, 2 * dim_model)

        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, t_emb: torch.Tensor):
        gamma, beta = self.modulation(t_emb).chunk(2, dim=-1)

        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]

        return gamma, beta

class FrequencySelfAttention(nn.Module):
    def __init__(self, dim_model: int = 256, n_heads: int = 4, dropout=0.1):
        super().__init__()
        self.dim_model = dim_model
        self.norm = nn.LayerNorm(dim_model)
        self.attention = nn.MultiheadAttention(
            dim_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gamma, beta):
        B, L, F, D = x.shape
        assert D == self.dim_model, f"{self.__class__.__name__}.dim_model != x dim"

        x_norm = self.norm(x)
        x_norm = x_norm * (1 + gamma) + beta
        x_flat = x_norm.reshape(B * L, F, D)

        attn_out, _ = self.attention(
            x_flat,
            x_flat,
            x_flat,
            need_weights=False,
        )

        attn_out = attn_out.reshape(B, L, F, D)
        return x + self.dropout(attn_out)

class SlotSelfAttention(nn.Module):
    def __init__(self, dim_model: int = 256, n_heads: int = 4, dropout=0.1):
        super().__init__()
        self.dim_model = dim_model
        self.norm = nn.LayerNorm(dim_model)
        self.attention = nn.MultiheadAttention(
            dim_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gamma, beta):
        B, L, F, D = x.shape
        assert D == self.dim_model, f"{self.__class__.__name__}.dim_model != x dim"

        x_norm = self.norm(x)
        x_norm = x_norm * (1 + gamma) + beta
        x_flat = x_norm.permute(0, 2, 1, 3).reshape(B * F, L, D)

        attn_out, _ = self.attention(
            x_flat,
            x_flat,
            x_flat,
            need_weights=False,
        )

        attn_out = attn_out.reshape(B, F, L, D).permute(0, 2, 1, 3)
        return x + self.dropout(attn_out)

class CrossAttentionToMixture(nn.Module):
    def __init__(self, dim_model: int = 256, n_heads: int = 4, dropout=0.1):
        super().__init__()
        self.dim_model = dim_model
        self.x_norm = nn.LayerNorm(dim_model)
        self.y_norm = nn.LayerNorm(dim_model)
        self.attention = nn.MultiheadAttention(
            dim_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, y, gamma, beta):
        # x: (B, L, F, D)
        # y: (B, F, D)

        B, L, F, D = x.shape
        By, Fy, Dy = y.shape

        assert By == B
        assert Fy == F
        assert D == self.dim_model
        assert Dy == self.dim_model

        x_norm = self.x_norm(x)
        x_norm = x_norm * (1 + gamma) + beta
        y_norm = self.y_norm(y)
        
        x_flat = x_norm.reshape(B, L * F, D)

        attn_out, _ = self.attention(
            x_flat,
            y_norm,
            y_norm,
            need_weights=False
        )
        attn_out = attn_out.reshape(B, L, F, D)

        return x + self.dropout(attn_out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim_model: int = 256,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim_model)

        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gamma, beta):
        x_norm = self.norm(x)
        x_norm = x_norm * (1 + gamma) + beta
        return x + self.dropout(self.ffn(x_norm))

class AxialSeparatorBlock(nn.Module):
    def __init__(
        self,
        dim_model: int = 256,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.film = FiLM(dim_model)

        self.freq_attn = FrequencySelfAttention(
            dim_model=dim_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.slot_attn = SlotSelfAttention(
            dim_model=dim_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.cross_attn = CrossAttentionToMixture(
            dim_model=dim_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.ffn = FeedForward(
            dim_model=dim_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x, y, t_emb):
        gamma, beta = self.film(t_emb)
        x = self.freq_attn(x, gamma, beta)
        x = self.slot_attn(x, gamma, beta)
        x = self.cross_attn(x, y, gamma, beta)
        x = self.ffn(x, gamma, beta)
        return x
    
if __name__ == "__main__":
    h = torch.randn(32, 3, 128, 256)
    print(f"h shape: {h.shape}")
    y = torch.randn(32, 128, 256)
    print(f"y shape: {y.shape}")
    t_emb = torch.randn(32, 256)

    film = FiLM()
    gamma, beta = film(t_emb)

    freq_attn = FrequencySelfAttention()
    out_freq = freq_attn(h, gamma, beta)
    print(f"out_freq.shape: {out_freq.shape}")

    slot_attn = SlotSelfAttention()
    out_slot = slot_attn(h, gamma, beta)
    print(f"out_slot.shape: {out_slot.shape}")

    cross_attn = CrossAttentionToMixture()
    out_cross = cross_attn(h, y, gamma, beta)
    print(f"out_cross.shape: {out_cross.shape}")


    block_1 = AxialSeparatorBlock()
    nb_params = sum(p.numel() for p in block_1.parameters() if p.requires_grad)
    print(f"nb_params : {nb_params}")
    out_1 = block_1(h, y, t_emb)
    print(f"out_1 shape: {out_1.shape}")

