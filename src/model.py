import torch
import torch.nn as nn
from src.modules import ConvEncoder, TimeEmbedding, AxialSeparatorBlock

class FlowSeparator(nn.Module):
    def __init__(self, input_channels: int = 4, max_k: int = 2, n_blocks: int = 4, dim_model: int = 256, n_heads: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()

        self.mixture_encoder = ConvEncoder(input_channels, dim_model, dropout)
        self.slot_encoder = ConvEncoder(input_channels, dim_model, dropout)

        self.freq_emb = nn.Embedding(128, dim_model)
        self.type_emb = nn.Embedding(2, dim_model) # either sources or residual
        self.k_emb = nn.Embedding(max_k + 1, dim_model) # +1 because it creates n elements starting from 0 
        self.t_emb = TimeEmbedding(dim_model)

        self.blocks = nn.ModuleList([
            AxialSeparatorBlock(dim_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_blocks)
        ])

        self.out = nn.Linear(dim_model, 4)

    def forward(self, 
                x_t : torch.Tensor, 
                t : torch.Tensor, 
                y : torch.Tensor,
                K : torch.Tensor,) -> torch.Tensor:
        # X_t: (B, L, 4, F) where F = 128 and L = K + 1
        # t:   (B,) 
        # y:   (B, 4, F) where F = 128
        # K:   (B,) 
        B, L, C, F = x_t.shape
        y_tok = self.mixture_encoder(y)  # (B, dim_model, F)
        y_tok = y_tok.permute(0, 2, 1)  # (B, F, dim_model)

        x_tok = self.slot_encoder(x_t) # (B, L, dim_model, F)
        x_tok = x_tok.permute(0, 1, 3, 2)  # (B, L, F, dim_model)

        freq_emb = self.freq_emb(torch.arange(F, dtype=torch.long, device=x_t.device))  # (B, F, dim_model)

        type_ids = torch.zeros(L, dtype=torch.long, device=x_t.device)
        type_ids[-1] = 1
        type_emb = self.type_emb(type_ids) # (B, L, dim_model)
        
        k_emb = self.k_emb(K) # (B, dim_model)
        t = t.reshape(B, 1) # (B, 1)
        t_emb = self.t_emb(t) # (B, dim_model)

        h = x_tok + freq_emb[None, None, :, :] + type_emb[None, :, None, :] + k_emb[:, None, None, :]

        for block in self.blocks:
            h = block(h, y_tok, t_emb) # (B, L, F, dim_model)

        v_raw = self.out(h)               # (B, L, F, 4)
        v_raw = v_raw.permute(0, 1, 3, 2) # (B, L, 4, F)

        v = v_raw - v_raw.mean(dim=1, keepdim=True)

        return v

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def interpolate_x(X_0, X_1, t): 
        t_view = t[:, None, None, None]
        return (1 - t_view) * X_0 + t_view * X_1

    B = 32
    K_val = 2
    K = torch.full((B,), K_val, dtype=torch.long, device=device)
    L = K_val + 1
    nb_of_channels = 4
    nb_of_freq_bins = 128

    S = torch.randn(B, L, nb_of_channels, nb_of_freq_bins, device=device) # (B, L, C, F)
    y = S.sum(dim=1) # (B, C, F)
    print(f"S shape : {S.shape}")
    print(f"y shape : {y.shape}")
    
    s_bar = y / L # (B, C, F)
    S_bar = s_bar.unsqueeze(1).repeat(1, L, 1, 1).to(device) # (B, L, C, F)
    print(f"s_bar shape : {s_bar.shape}")
    print(f"S_bar shape : {S_bar.shape}")

    Z = torch.randn(B, L, nb_of_channels, nb_of_freq_bins, device=device)
    Z = Z - Z.mean(dim=1, keepdim=True)
    print(f"Z shape : {Z.shape}")

    X_0 = S_bar + Z
    X_1 = S
    t = torch.rand(B, device=device)

    X_t = interpolate_x(X_0, X_1, t)
    print(f"X_t shape : {X_t.shape}")

    model = FlowSeparator(input_channels=nb_of_channels, max_k=K_val, n_blocks=4, dim_model=256, n_heads=4, dim_feedforward=512, dropout=0.1).to(device)
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {nb_params}")
    out = model(X_t, t, y, K)
    print(f"out shape : {out.shape}")