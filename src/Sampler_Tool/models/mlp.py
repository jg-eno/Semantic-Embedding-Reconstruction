import torch.nn as nn


class SingleTokenMLP(nn.Module):
    """Projects a paragraph embedding to one decoder hidden-state position (soft prefix token)."""

    def __init__(self, paragraph_dim: int, decoder_hidden_dim: int, mlp_hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(paragraph_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, decoder_hidden_dim),
        )

    def forward(self, x):
        return self.net(x)
