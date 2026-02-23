# model_transformer_window.py
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, L, D = x.shape
        positions = torch.arange(0, L, device=x.device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb

class TransformerSleepModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, n_heads=8, num_layers=3, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.shared_encoder = self.transformer_encoder

        self.arousal_classifier = nn.Linear(hidden_dim, 2)
        self.respiratory_classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.input_proj(x)           # (B, W, H)
        x = self.pos_encoder(x)          # (B, W, H)
        x = self.shared_encoder(x)       # (B, W, H)

        center_idx = x.size(1) // 2
        center_token = x[:, center_idx, :]   # (B, H)

        return (
            self.arousal_classifier(center_token),
            self.respiratory_classifier(center_token)
        )