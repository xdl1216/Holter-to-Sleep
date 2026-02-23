import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # Ensure x has 3 dimensions (B, L, D)
        if x.dim() != 3:
            raise ValueError(f"Expected input to have 3 dimensions, but got {x.dim()} dimensions")
        B, L, D = x.shape  # Unpack into batch size, sequence length, and feature dimension
        positions = torch.arange(0, L, device=x.device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb

class TransformerSleepModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_heads=8, num_layers=3, num_classes=5, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)  # Adjust input_dim for window_size * feature_dim
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)  # For Sleep Staging (5 classes)

    def forward(self, x):
        # Flatten the window size and feature dimensions
        x = x.view(x.size(0), x.size(1), -1)  # Flatten window size and feature_dim to a single dimension
        
        # Apply the linear projection
        x = self.input_proj(x)  # (B, L, H)
        x = self.pos_encoder(x)  # (B, L, H)
        x = self.transformer_encoder(x)  # (B, L, H)
        sleep_logits = self.classifier(x)  # (B, L, num_classes)

        return sleep_logits

