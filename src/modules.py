import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

import math
from collections import OrderedDict

from src.masking import mask_slots

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
    def __init__(self, dim_model : int = 256, n_heads : int = 4, dropout : float = 0.1):
        super().__init__()
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.dropout_p = dropout

        self.head_dim = dim_model // n_heads

        self.qkv_proj = nn.Linear(dim_model, dim_model * 3)
        self.out_proj = nn.Linear(dim_model, dim_model)
        
        self.norm = nn.RMSNorm(dim_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gamma, beta, slot_mask):
        B, L, F, D = x.shape # B = Batch Size, L = Number of Slots, F = Number of frequency bins, D = dim_model
        assert D == self.dim_model, f"{self.__class__.__name__}.dim_model != x_dim"

        x_norm = self.norm(x)
        x_norm = x_norm * (1 + gamma) + beta

        qkv = self.qkv_proj(x_norm) # (B*L, F, 3*D)
        q, k, v = torch.chunk(input=qkv, chunks=3, dim=-1) # each of shape (B, L, F, D)

        q = q.view(B, L, F, self.n_heads, self.head_dim).transpose(2, 3)
        k = k.view(B, L, F, self.n_heads, self.head_dim).transpose(2, 3)
        v = v.view(B, L, F, self.n_heads, self.head_dim).transpose(2, 3)

        dropout = self.dropout_p if self.training else 0.0
        attn_out = scaled_dot_product_attention(query = q, 
                                                key = k , 
                                                value = v, 
                                                dropout_p = dropout
                                                ) # shape : B*L, F, n_heads, head_dim
        
        attn_out = attn_out.transpose(2, 3).reshape(B, L, F, self.dim_model)
        out = self.out_proj(attn_out)

        return mask_slots(x + self.dropout(out), slot_mask)

class SlotSelfAttention(nn.Module):
    def __init__(self, dim_model: int = 256, n_heads: int = 4, dropout : float = 0.1):
        super().__init__()
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.dropout_p = dropout

        self.head_dim = dim_model // n_heads

        self.qkv_proj = nn.Linear(dim_model, dim_model * 3)
        self.out_proj = nn.Linear(dim_model, dim_model)

        self.norm = nn.RMSNorm(dim_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gamma, beta, slot_mask):
        B, L, F, D = x.shape
        assert D == self.dim_model, f"{self.__class__.__name__}.dim_model != x dim"

        x_norm = self.norm(x)
        x_norm = x_norm * (1 + gamma) + beta

        qkv = self.qkv_proj(x_norm)
        q, k, v = torch.chunk(input = qkv, chunks = 3, dim = -1) # each of shape (B, L, F, D)

        q = q.permute(0, 2, 1, 3).view(B, F, L, self.n_heads, self.head_dim).transpose(2, 3)
        k = k.permute(0, 2, 1, 3).view(B, F, L, self.n_heads, self.head_dim).transpose(2, 3)
        v = v.permute(0, 2, 1, 3).view(B, F, L, self.n_heads, self.head_dim).transpose(2, 3)

        dropout = self.dropout_p if self.training else 0.0

        attn_out = scaled_dot_product_attention(
            query = q, 
            key = k,
            value = v,
            attn_mask=slot_mask[:, None, None, None, :],
            dropout_p = dropout
        ) # shape : (B, L, n_heads, F, head_dim)

        attn_out = attn_out.transpose(2, 3).reshape(B, F, L, self.dim_model) # shape : (B, F, L, D)
        out = self.out_proj(attn_out)
        out = out.permute(0, 2, 1, 3)

        return mask_slots(x + self.dropout(out), slot_mask)

class CrossAttentionToMixture(nn.Module):
    def __init__(self, dim_model: int = 256, n_heads: int = 4, dropout : float = 0.1):
        super().__init__()
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.dropout_p = dropout

        self.head_dim = dim_model // n_heads

        self.query_proj = nn.Linear(dim_model, dim_model)
        self.kv_proj = nn.Linear(dim_model, 2 * dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

        self.x_norm = nn.RMSNorm(dim_model, eps=1e-6)
        self.y_norm = nn.RMSNorm(dim_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, y, gamma, beta, slot_mask):
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

        q = self.query_proj(x_norm)  # (B, L, F, D)
        kv = self.kv_proj(y_norm)  # (B, F, 2 * D)
        k, v = kv.chunk(2, dim=-1)  # each has shape (B, F, D)

        q = q.reshape(B, L * F, self.n_heads, self.head_dim).transpose(1, 2) # shape : (B, n_heads, L * F, head_dim)
        k = k.reshape(B, F, self.n_heads, self.head_dim).transpose(1, 2) # shape : (B, n_heads, F, head_dim)
        v = v.reshape(B, F, self.n_heads, self.head_dim).transpose(1, 2) # shape : (B, n_heads, F, head_dim)

        dropout = self.dropout_p if self.training else 0.0

        attn_out = scaled_dot_product_attention(
            query = q,
            key = k,
            value= v,
            dropout_p=dropout
        ) # shape : (B, n_heads, L * F, head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(B, L, F, D)

        out = self.out_proj(attn_out)

        return mask_slots(x + self.dropout(out), slot_mask)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim_model: int = 256,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm = nn.RMSNorm(dim_model, eps=1e-6)

        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gamma, beta, slot_mask):
        x_norm = self.norm(x)
        x_norm = x_norm * (1 + gamma) + beta
        return mask_slots(x + self.dropout(self.ffn(x_norm)), slot_mask)

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

    def forward(self, x, y, t_emb, slot_mask):
        gamma, beta = self.film(t_emb)
        x = self.freq_attn(x, gamma, beta, slot_mask)
        x = self.slot_attn(x, gamma, beta, slot_mask)
        x = self.cross_attn(x, y, gamma, beta, slot_mask)
        x = self.ffn(x, gamma, beta, slot_mask)
        return x
    
if __name__ == "__main__":
    B, L, F, D = 512, 11, 128, 256
    h = torch.randn(B, L, F, D)
    print(f"h shape: {h.shape}")
    y = torch.randn(B, F, D)
    print(f"y shape: {y.shape}")
    t_emb = torch.randn(B, D)

    film = FiLM()
    gamma, beta = film(t_emb)

    freq_attn = FrequencySelfAttention()
    slot_mask = torch.ones(B, L, dtype=torch.bool)

    out_freq = freq_attn(h, gamma, beta, slot_mask)
    print(f"out_freq.shape: {out_freq.shape}")

    slot_attn = SlotSelfAttention()
    out_slot = slot_attn(h, gamma, beta, slot_mask)
    print(f"out_slot.shape: {out_slot.shape}")

    cross_attn = CrossAttentionToMixture()
    out_cross = cross_attn(h, y, gamma, beta, slot_mask)
    print(f"out_cross.shape: {out_cross.shape}")


    block_1 = AxialSeparatorBlock()
    nb_params = sum(p.numel() for p in block_1.parameters() if p.requires_grad)
    print(f"nb_params : {nb_params}")
    out_1 = block_1(h, y, t_emb, slot_mask)
    print(f"out_1 shape: {out_1.shape}")
